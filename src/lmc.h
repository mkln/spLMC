#include <RcppArmadillo.h>
#include "daggp.h"
#include "ramadapt.h"
#include "diwishlog.h"

class LMC {
public:
  // A: q x q. theta: 4 x q with rows (phi, sigmasq, nu, tausq). gps: size q.
  LMC(const arma::mat& Y,
    const arma::mat& coords, const arma::mat& A, 
      const arma::mat& theta_opts, const arma::field<arma::uvec>& dag, 
      int dagopts, int nthreads)
    : Y_(Y), q(A.n_rows), A_(A)
  {
    if (A_.n_cols != q) Rcpp::stop("A must be square q x q");
    
    A_ = arma::chol(A_ * A_.t(), "lower"); // identifiable
    
    n = coords.n_rows;                         // initialize n_
    matern = true;
    
    theta_options = theta_opts;
    n_options = theta_options.n_cols;
    daggp_options = std::vector<DagGP>(n_options);//.reserve(n_options);
    
    // if multiple nu options, interpret as wanting to sample smoothness for matern
    // otherwise, power exponential with fixed exponent.
    phi_sampling = (q == 1) | (arma::var(theta_options.row(0)) != 0);
    sigmasq_sampling = (q == 1) | (arma::var(theta_options.row(1)) != 0);
    nu_sampling = (q == 1) | (arma::var(theta_options.row(2)) != 0);
    tausq_sampling = (q == 1) | (arma::var(theta_options.row(3)) != 0);
    
    for(unsigned int i=0; i<q; i++){
      daggp_options[i] = DagGP(coords, theta_options.col(i), dag, 
                               dagopts,
                               true, 
                               0, // with q blocks, make Ci
                               nthreads);
    }
    daggp_options_alt = daggp_options;
    
    init_adapt();
  }
  
  // Update hyperparameters from a 4 x q matrix.
  void update_theta(const arma::mat& theta) {
    if (theta.n_rows != 4 || theta.n_cols != q) Rcpp::stop("theta must be 4 x q");
    for (std::size_t j = 0; j < q; ++j) {
      daggp_options[j].update_theta(theta.col(j), true);
    }
  }
  
  double loglik(const arma::mat& Y) const {
    const std::size_t N = n*q;
    
    arma::mat At = A_.t();
    arma::mat AinvT_ = arma::solve(At, arma::eye(q, q));  // A^T X = I  -> X = A^{-T}
    
    // 1) u = (A^{-1} ⊗ I_n) y  == vec( Y * A^{-T} )
    arma::mat W = Y * AinvT_;        // n x q
    arma::vec u = arma::vectorise(W);
    
    // 2) t = H u, where H = blkdiag(H_1,...,H_q); H_j upper-triangular
    arma::vec t(N, arma::fill::zeros);
    double logdetR = 0.0;            // accumulate ∑ log|R_j|
    for (std::size_t j = 0; j < q; ++j) {
      logdetR += -daggp_options[j].precision_logdeterminant;
      // t_j = Hj * u_j  (block of length n)
      arma::uvec idx = arma::regspace<arma::uvec>(j*n, (j+1)*n - 1);
      //arma::mat u_mat = u(idx);
      t(idx) = daggp_options[j].H * u(idx);
    }
    double quad = arma::dot(t, t);
    
    double val, sign;
    arma::log_det(val, sign, A_);
    if (sign <= 0.0) Rcpp::stop("A must have positive determinant");
    double logdetA_ = val;
    
    // 3) det term
    double logdetSigma = 2.0 * static_cast<double>(n) * logdetA_ + logdetR;
    
    // 4) final log-likelihood
    const double cst = static_cast<double>(N) * std::log(2.0 * M_PI);
    return -0.5 * (cst + logdetSigma + quad);
  }
  
  double loglik_curr() const {
    const std::size_t N = n*q;
    
    arma::mat At = arma::chol(A_ * A_.t(), "upper");
    arma::mat AinvT_ = arma::solve(At, arma::eye(q, q));  // A^T X = I  -> X = A^{-T}
    
    // 1) u = (A^{-1} ⊗ I_n) y  == vec( Y * A^{-T} )
    arma::mat W = Y_ * AinvT_;        // n x q
    arma::vec u = arma::vectorise(W);
    
    // 2) t = H u, where H = blkdiag(H_1,...,H_q); H_j upper-triangular
    arma::vec t(N, arma::fill::zeros);
    double logdetR = 0.0;            // accumulate ∑ log|R_j|
    for (std::size_t j = 0; j < q; ++j) {
      logdetR += -daggp_options[j].precision_logdeterminant;
      // t_j = Hj * u_j  (block of length n)
      arma::uvec idx = arma::regspace<arma::uvec>(j*n, (j+1)*n - 1);
      //arma::mat u_mat = u(idx);
      t(idx) = daggp_options[j].H * u(idx);
    }
    double quad = arma::dot(t, t);
    
    // 3) det term
    double val, sign;
    arma::log_det(val, sign, At);
    if (sign <= 0.0) Rcpp::stop("A must have positive determinant");
    double logdetA_ = val;
    
    double logdetSigma = 2.0 * static_cast<double>(n) * logdetA_ + logdetR;
    
    // 4) final log-likelihood
    const double cst = static_cast<double>(N) * std::log(2.0 * M_PI);
    return -0.5 * (cst + logdetSigma + quad);
  }
  
  double loglik_theta_prop() const {
    const std::size_t N = n*q;
    
    arma::mat At = arma::chol(A_ * A_.t(), "upper");
    arma::mat AinvT_ = arma::solve(At, arma::eye(q, q));  // A^T X = I  -> X = A^{-T}
  
    // 1) u = (A^{-1} ⊗ I_n) y  == vec( Y * A^{-T} )
    arma::mat W = Y_ * AinvT_;        // n x q
    arma::vec u = arma::vectorise(W);
    
    // 2) t = H u, where H = blkdiag(H_1,...,H_q); H_j upper-triangular
    arma::vec t(N, arma::fill::zeros);
    double logdetR = 0.0;            // accumulate ∑ log|R_j|
    for (std::size_t j = 0; j < q; ++j) {
      logdetR += -daggp_options_alt[j].precision_logdeterminant;
      // t_j = Hj * u_j  (block of length n)
      arma::uvec idx = arma::regspace<arma::uvec>(j*n, (j+1)*n - 1);
      //arma::mat u_mat = u(idx);
      t(idx) = daggp_options_alt[j].H * u(idx);
    }
    double quad = arma::dot(t, t);
    
    double val, sign;
    arma::log_det(val, sign, At);
    if (sign <= 0.0) Rcpp::stop("A must have positive determinant");
    double logdetA_ = val;
    
    // 3) det term
    double logdetSigma = 2.0 * static_cast<double>(n) * logdetA_ + logdetR;
    
    // 4) final log-likelihood
    const double cst = static_cast<double>(N) * std::log(2.0 * M_PI);
    return -0.5 * (cst + logdetSigma + quad);
  }
  
  double loglik_A_prop(const arma::mat& A_alt) const {
    
    arma::mat At = arma::chol(A_alt * A_alt.t(), "upper");
    arma::mat AinvT_ = arma::solve(At, arma::eye(q, q));  // A^T X = I  -> X = A^{-T}
    
    const std::size_t N = n*q;
    
    // 1) u = (A^{-1} ⊗ I_n) y  == vec( Y * A^{-T} )
    arma::mat W = Y_ * AinvT_;        // n x q
    arma::vec u = arma::vectorise(W);
    
    // 2) t = H u, where H = blkdiag(H_1,...,H_q); H_j upper-triangular
    arma::vec t(N, arma::fill::zeros);
    double logdetR = 0.0;            // accumulate ∑ log|R_j|
    for (std::size_t j = 0; j < q; ++j) {
      logdetR += -daggp_options[j].precision_logdeterminant;
      // t_j = Hj * u_j  (block of length n)
      arma::uvec idx = arma::regspace<arma::uvec>(j*n, (j+1)*n - 1);
      //arma::mat u_mat = u(idx);
      t(idx) = daggp_options[j].H * u(idx);
    }
    double quad = arma::dot(t, t);
    
    double val, sign;
    arma::log_det(val, sign, At);
    if (sign <= 0.0) Rcpp::stop("A must have positive determinant");
    double logdetA_alt = val;
    
    // 3) det term
    double logdetSigma = 2.0 * static_cast<double>(n) * logdetA_alt + logdetR;
    
    // 4) final log-likelihood
    const double cst = static_cast<double>(N) * std::log(2.0 * M_PI);
    return -0.5 * (cst + logdetSigma + quad);
  }
  
  std::size_t n_sites() const { return n; }
  std::size_t q_dim()   const { return q; }
  
  arma::mat Y_;
  std::size_t q, n;
  arma::mat A_, A_alt;
  double logdetA_;
  std::vector<DagGP> daggp_options, daggp_options_alt;
  arma::mat theta_options; // each column is one alternative value for theta
  unsigned int n_options;
  
  bool matern;
  
  bool phi_sampling, sigmasq_sampling, nu_sampling, tausq_sampling;
  // adaptive metropolis to update theta atoms
  arma::uvec which_theta_elem;
  
  // adaptive metropolis (conditional update) to update theta atoms
  // assume shared covariance functions and unknown parameters across variables
  arma::mat c_theta_unif_bounds;
  std::vector<RAMAdapt> c_theta_adapt;
  int theta_mcmc_counter;
  
  void init_adapt();
  void upd_thetaj_metrop();
  void upd_A_metrop();
  
  arma::mat A_unif_bounds;
  RAMAdapt A_adapt;
  int A_mcmc_counter;
};

inline void LMC::init_adapt(){
  // adaptive metropolis
  theta_mcmc_counter = 0;
  which_theta_elem = arma::zeros<arma::uvec>(0);
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  
  if(phi_sampling){
    which_theta_elem = arma::join_vert(which_theta_elem, 0*oneuv);
  }
  if(sigmasq_sampling){
    which_theta_elem = arma::join_vert(which_theta_elem, 1*oneuv);
  }
  if(nu_sampling){
    which_theta_elem = arma::join_vert(which_theta_elem, 2*oneuv);
  }
  if(tausq_sampling){
    which_theta_elem = arma::join_vert(which_theta_elem, 3*oneuv);
  }
  
  arma::mat bounds_all = arma::zeros(4, 2); // make bounds for all, then subset
  bounds_all.row(0) = arma::rowvec({.3, 100}); // phi
  bounds_all.row(1) = arma::rowvec({1e-6, 100}); // sigma
  if(matern){
    bounds_all.row(2) = arma::rowvec({0.49, 2.1}); // nu  
  } else {
    // power exponential
    bounds_all.row(2) = arma::rowvec({1, 2}); // nu
  }
  
  bounds_all.row(3) = arma::rowvec({1e-16, 100}); // tausq
  bounds_all = bounds_all.rows(which_theta_elem);
  
  //if(n_options == q){
  // conditional update
  c_theta_unif_bounds = bounds_all;
  int c_theta_par = which_theta_elem.n_elem;
  arma::mat c_theta_metrop_sd = 0.05 * arma::eye(c_theta_par, c_theta_par);
  c_theta_adapt = std::vector<RAMAdapt>(n_options);
  for(int j=0; j<n_options; j++){
    c_theta_adapt[j] = RAMAdapt(c_theta_par, c_theta_metrop_sd, 0.24);
  }
  // ---  
  //}
  
  Rcpp::Rcout << "A unif bounds \n";
  
  // metropolis for A
  int n_A_par = q*q;
  A_unif_bounds = arma::zeros(q*q, 2);
  A_unif_bounds.col(0) += -1e10;
  A_unif_bounds.col(1) += 1e10;
  
  arma::mat A_metrop_sd = 0.05 * arma::eye(n_A_par, n_A_par);
  A_adapt = RAMAdapt(n_A_par, A_metrop_sd, 0.24);
  //A_adapt_active = true;
}

inline void LMC::upd_thetaj_metrop(){
  
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  for(int j=0; j<q; j++){
    daggp_options_alt = daggp_options;
    c_theta_adapt[j].count_proposal();
    
    arma::vec phisig_cur = theta_options(which_theta_elem, oneuv*j);
    
    Rcpp::RNGScope scope;
    arma::vec U_update = arma::randn(phisig_cur.n_elem);
    
    arma::vec phisig_alt = par_huvtransf_back(par_huvtransf_fwd(
      phisig_cur, c_theta_unif_bounds) + 
        c_theta_adapt[j].paramsd * U_update, c_theta_unif_bounds);
    
    // proposal for theta matrix
    arma::mat theta_alt = theta_options;
    theta_alt(which_theta_elem, oneuv*j) = phisig_alt; 
    
    if(!theta_alt.is_finite()){
      Rcpp::stop("Some value of theta outside of MCMC search limits.\n");
    }
    
    // ---------------------
    // create proposal daggp
    daggp_options_alt[j].update_theta(theta_alt.col(j), true);
    
    double curr_logdens = loglik_curr();
    double prop_logdens = loglik_theta_prop();
      
    // priors
    double logpriors = 0;
    if(sigmasq_sampling){
      logpriors += invgamma_logdens(theta_alt(1,j), 2, 1) - invgamma_logdens(theta_options(1,j), 2, 1);
    }
    if(tausq_sampling){
      logpriors += expon_logdens(theta_alt(3,j), 25) - expon_logdens(theta_options(3,j), 25);
    }
    
    // ------------------
    // make move
    double jacobian  = calc_jacobian(phisig_alt, phisig_cur, c_theta_unif_bounds);
    double logaccept = prop_logdens - curr_logdens + jacobian + logpriors;
    
    bool accepted = do_I_accept(logaccept);
    
    if(accepted){
      theta_options = theta_alt;
      std::swap(daggp_options.at(j), daggp_options_alt.at(j));
    } 
    
    c_theta_adapt[j].update_ratios();
    
    //if(theta_adapt_active){
      c_theta_adapt[j].adapt(U_update, exp(logaccept), theta_mcmc_counter); 
    //}
    
    theta_mcmc_counter++;
  }
  
}

inline void LMC::upd_A_metrop(){
  
  arma::uvec oneuv = arma::ones<arma::uvec>(1);

  A_adapt.count_proposal();
  
  arma::vec Av_cur = arma::vectorise(A_);
  
  Rcpp::RNGScope scope;
  arma::vec U_update = arma::randn(Av_cur.n_elem);
  
  arma::vec Av_alt = Av_cur + A_adapt.paramsd * U_update;
  
  // proposal for theta matrix
  arma::mat A_alt = arma::mat(Av_alt.memptr(), q, q); 
  
  //Rcpp::Rcout << A_adapt.paramsd << endl;
  //Rcpp::Rcout << "A current \n" << A_ << endl;
  //Rcpp::Rcout << "A propose \n" << A_alt << endl; 
  
  if(!A_alt.is_finite()){
    Rcpp::stop("Some value of theta outside of MCMC search limits.\n");
  }
  
  double curr_logdens = loglik_A_prop(A_);
  double prop_logdens = loglik_A_prop(A_alt);
  
  arma::mat iwishS = arma::eye(q,q);
  int v = iwishS.n_cols+1;
  
  arma::mat Sigma_ = A_ * A_.t();
  arma::mat Sigma_alt = A_alt * A_alt.t();
  double logprior_cur = log_iwish_density(Sigma_, iwishS, v);
  double logprior_alt = log_iwish_density(Sigma_alt, iwishS, v);
  
  // ------------------
  // make move
  //double jacobian  = calc_jacobian(Av_alt, Av_cur, A_unif_bounds);
  double logaccept = prop_logdens - curr_logdens + //jacobian + 
    logprior_cur - logprior_alt;
  
  //Rcpp::Rcout << "current: " << curr_logdens << " -> prop: " << prop_logdens << 
  //  "\n prior current: " << logprior_cur << " --> prop: " << logprior_alt << 
  //    "\n jac: " << jacobian << endl;
  
  bool accepted = do_I_accept(logaccept);
  
  if(accepted){
    //Rcpp::Rcout << "accepted" << endl;
    A_ = A_alt;
  } 
  
  A_adapt.update_ratios();
  
  //if(theta_adapt_active){
  A_adapt.adapt(U_update, exp(logaccept), A_mcmc_counter); 
  //}
  A_mcmc_counter++;
}
