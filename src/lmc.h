#include <RcppArmadillo.h>
#include "daggp.h"
#include "ramadapt.h"
#include "diwishlog.h"


inline arma::mat t_fix_diag_sign(const arma::mat& A) {
  arma::mat Astar = A;
  int q = A.n_rows;
  for (int j = 0; j < q; ++j) {
    if (Astar(j,j) < 0) {
      Astar.col(j) *= -1.0;
    }
  }
  return Astar.t();
}


class LMC {
public:
  // A: q x q. theta: 4 x q with rows (phi, sigmasq, nu, tausq). gps: size q.
  LMC(const arma::mat& Y,
    const arma::mat& coords, const arma::mat& A, 
      const arma::mat& theta_opts, const arma::field<arma::uvec>& dag, 
      int dagopts, int nthreads, bool cov_matern = true)
    : Y_(Y), q(A.n_rows), k(theta_opts.n_cols), A_(A), n_threads(nthreads)
  {
    

    n = coords.n_rows;                         // initialize n_
    matern = cov_matern;
    
    
    // if multiple nu options, interpret as wanting to sample smoothness for matern
    // otherwise, power exponential with fixed exponent.
    phi_sampling = (q == 1) | (arma::var(theta_opts.row(0)) != 0);
    sigmasq_sampling = (q == 1) | (arma::var(theta_opts.row(1)) != 0);
    nu_sampling = (q == 1) | (arma::var(theta_opts.row(2)) != 0);
    tausq_sampling = (q == 1) | (arma::var(theta_opts.row(3)) != 0);
    
    if(k < q){
      arma::mat add_noise = arma::zeros(4, q-k);
      add_noise.row(0) += 1e10;
      add_noise.row(1) += 1;
      add_noise.row(2) += 0.5;
      theta_options = arma::join_horiz(theta_opts, add_noise);
    } else {
      theta_options = theta_opts;
    }
    
    daggp_options = std::vector<DagGP>(q);
    
    for(unsigned int i=0; i<q; i++){
      daggp_options[i] = DagGP(coords, theta_options.col(i), dag, 
                               dagopts,
                               cov_matern, 
                               0, // with q blocks, make Ci
                               nthreads);
    }
    //daggp_options_alt = daggp_options;
    
    init_adapt();
    
    U = arma::zeros(n, q);
    V = arma::zeros(n, q);
    
    build_u_cache_();
    build_H_cache_();
    
    logdens = logdens_curr_fast_();
  }
  
  int n_threads;
  
  double logdens;
  
  double logdens_A_prop(const arma::mat& A_alt) {
    
    arma::mat At = t_fix_diag_sign(A_alt);
    arma::mat AinvT_ = arma::solve(At, arma::eye(q, q)); 
    
    const std::size_t N = n*q;
    
    U = Y_ * AinvT_;       
    double logdetR = 0.0;            
    for (std::size_t j = 0; j < q; ++j) {
      logdetR += -daggp_options[j].precision_logdeterminant;
      V.col(j) = daggp_options[j].H * U.col(j);
    }
    double quad = arma::accu(V % V);
    
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
  
  // compute prop logdens for index j only (no side effects)
  double logdens_theta_prop_single_(int j, const DagGP& gp_prop) const {
    const double quad_curr = arma::accu(V % V);
    
    arma::vec V_prop = gp_prop.H * U.col(j);
    const double quad_prop = quad_curr
    - arma::dot(V.col(j), V.col(j))
      + arma::dot(V_prop, V_prop);
    
    const double logdetR_prop = logdetR_sum_
    - (-daggp_options[j].precision_logdeterminant)
      + (-gp_prop.precision_logdeterminant);
  
    const double logdetSigma_prop = 2.0 * double(n) * logdetA_ + logdetR_prop;
    const double cst = double(n*q) * std::log(2.0 * M_PI);
    return -0.5 * (cst + logdetSigma_prop + quad_prop);
  }
  
  double logdens_curr_fast_() {
    logdetA_ = arma::accu(log(A_.diag()));
    
    const double quad_curr = arma::accu(V % V);
    const double logdetSigma_curr = 2.0 * double(n) * logdetA_ + logdetR_sum_;
    const double cst = double(n*q) * std::log(2.0 * M_PI);
    
    return -0.5 * (cst + logdetSigma_curr + quad_curr);
  }
  
  
  std::size_t n_sites() const { return n; }
  std::size_t q_dim()   const { return q; }
  
  arma::mat Y_;
  std::size_t q, n, k;
  arma::mat A_;
  double logdetA_;
  std::vector<DagGP> daggp_options;
  arma::mat theta_options; // each column is one alternative value for theta
  
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
  void sample_A_conditional();
  
  arma::mat A_unif_bounds;
  RAMAdapt A_adapt;
  int A_mcmc_counter;
  
  // caches
  arma::mat U;
  arma::mat V;                  
  double logdetR_sum_{0.0};
  bool caches_valid_{false};
  
  // U is Y whitened from cross-outcome dependence
  void build_u_cache_() {
    arma::mat At = t_fix_diag_sign(A_);
    arma::mat AinvT_ = arma::solve(At, arma::eye(q, q));  // A^T X = I  -> X = A^{-T}
    U = Y_ * AinvT_;
  }
  
  // V is U whitened from spatial dependence
  void build_H_cache_() {
    logdetR_sum_ = 0.0;
    for (std::size_t j=0; j<q; ++j) {
      V.col(j) = daggp_options[j].H * U.col(j);
      logdetR_sum_ += -daggp_options[j].precision_logdeterminant;
    }
    caches_valid_ = true;
  }
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
  bounds_all.row(0) = arma::rowvec({.3, 300}); // phi
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
  c_theta_adapt = std::vector<RAMAdapt>(k);
  for(int j=0; j<k; j++){
    c_theta_adapt[j] = RAMAdapt(c_theta_par, c_theta_metrop_sd, 0.24);
  }
  // ---  
  //}

  // metropolis for A
  int n_A_par = q*(q+1)/2;
  A_unif_bounds = arma::zeros(q*(q+1)/2, 2);
  A_unif_bounds.col(0) += -1e10;
  A_unif_bounds.col(1) += 1e10;
  
  arma::mat A_metrop_sd = 0.05 * arma::eye(n_A_par, n_A_par);
  A_adapt = RAMAdapt(n_A_par, A_metrop_sd, 0.24);
  //A_adapt_active = true;
}

inline void LMC::upd_thetaj_metrop() {
  if (!caches_valid_) { build_u_cache_(); build_H_cache_(); }
  
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  for (int j = 0; j < k; ++j) {     
    c_theta_adapt[j].count_proposal();
    
    arma::vec cur = theta_options(which_theta_elem, oneuv*j);
    Rcpp::RNGScope scope;
    arma::vec U_update = arma::randn(cur.n_elem);
    arma::vec alt = par_huvtransf_back(
      par_huvtransf_fwd(cur, c_theta_unif_bounds) +
        c_theta_adapt[j].paramsd * U_update,
        c_theta_unif_bounds);
    
    arma::mat theta_alt = theta_options;
    theta_alt(which_theta_elem, oneuv*j) = alt;
    if (!theta_alt.is_finite()) Rcpp::stop("theta out of bounds");
    
    // local temporary GP to avoid mutating others
    DagGP gp_prop = daggp_options[j];
    gp_prop.update_theta(theta_alt.col(j), true);
    
    const double curr_logdens = logdens_curr_fast_();
    const double prop_logdens = logdens_theta_prop_single_(j, gp_prop);
    
    double logpriors = 0.0;
    if (sigmasq_sampling)
      logpriors += invgamma_logdens(theta_alt(1,j), 2, 1) - invgamma_logdens(theta_options(1,j), 2, 1);
    if (tausq_sampling)
      logpriors += expon_logdens(theta_alt(3,j), 25) - expon_logdens(theta_options(3,j), 25);
    
    const double jac = calc_jacobian(alt, cur, c_theta_unif_bounds);
    const double logaccept = prop_logdens - curr_logdens + jac + logpriors;
    
    const bool accepted = do_I_accept(logaccept);
    if (accepted) {
      // commit θ_j
      theta_options = theta_alt;
      daggp_options[j] = gp_prop;
      
      // update caches for j only
      const arma::uword s = j*n, e = (j+1)*n - 1;
      // remove old contribution from aggregates
      logdetR_sum_ -= -daggp_options[j].precision_logdeterminant; // alt vector copied earlier; same as old
      logdetR_sum_ += -gp_prop.precision_logdeterminant;
      
      //const double old_q = arma::dot(t_blocks_[j], t_blocks_[j]);
      //const double all_q = arma::dot(t_, t_);
      // recompute new t_j and aggregates
      V.col(j) = gp_prop.H * U.col(j);
      
      // no need to recompute quad here; logdens_curr_fast_ will recompute dot(t_,t_) next time
      
      logdens = prop_logdens;
    } else {
      logdens = curr_logdens;
    }
    
    c_theta_adapt[j].update_ratios();
    c_theta_adapt[j].adapt(U_update, std::exp(std::min(0.0, logaccept)), theta_mcmc_counter);
    ++theta_mcmc_counter;
  }
}

inline arma::vec lower_triangular(const arma::mat& A) {
  int q = A.n_rows;
  int m = q * (q + 1) / 2;
  arma::vec out(m);
  int idx = 0;
  for (int j = 0; j < q; ++j) {
    for (int i = j; i < q; ++i) {
      out(idx++) = A(i, j);
    }
  }
  return out;
}

inline arma::mat lower_triangular_matrix(const arma::vec& v, int q) {
  arma::mat A(q, q, arma::fill::zeros);
  int idx = 0;
  for (int j = 0; j < q; ++j) {
    for (int i = j; i < q; ++i) {
      A(i, j) = v(idx++);
    }
  }
  return A;
}

inline void LMC::upd_A_metrop(){
  
  arma::uvec oneuv = arma::ones<arma::uvec>(1);

  A_adapt.count_proposal();
  
  arma::vec Av_cur = lower_triangular(A_);
  
  Rcpp::RNGScope scope;
  arma::vec U_update = arma::randn(Av_cur.n_elem);
  
  arma::vec Av_alt = Av_cur + A_adapt.paramsd * U_update;
  
  // proposal for theta matrix
  arma::mat A_alt = lower_triangular_matrix(Av_alt, q); 
  
  //Rcpp::Rcout << A_adapt.paramsd << endl;
  //Rcpp::Rcout << "A current \n" << A_ << endl;
  //Rcpp::Rcout << "A propose \n" << A_alt << endl; 
  
  if(!A_alt.is_finite()){
    Rcpp::stop("Some value of theta outside of MCMC search limits.\n");
  }
  
  double curr_logdens = logdens_A_prop(A_);
  double prop_logdens = logdens_A_prop(A_alt);
  
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
    
    build_u_cache_();
    build_H_cache_();
    
    logdens = prop_logdens;
  } else {
    logdens = curr_logdens;
  }
  
  A_adapt.update_ratios();
  
  //if(theta_adapt_active){
  A_adapt.adapt(U_update, exp(logaccept), A_mcmc_counter); 
  //}
  A_mcmc_counter++;
}

inline void LMC::sample_A_conditional(){
  
  arma::mat Bcond = arma::zeros(q, q);
  arma::vec sigmas = arma::zeros(q);
  
  arma::mat rand_mvn = arma::randn(q, q);
  
  sigmas(0) = sqrt( daggp_options[0].theta(1) );
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif
  for(int j=1; j<q; j++){
    sigmas(j) = sqrt( daggp_options[j].theta(1) );
    
    arma::mat X_local = Y_.cols(0, j-1);
    arma::vec y_local = Y_.col(j);
    
    arma::sp_mat Ci = daggp_options[j].H.t() * daggp_options[j].H;
    arma::mat XtCi = X_local.t() * Ci;
    arma::mat XtCiX = XtCi * X_local;
    arma::mat XtCiy = XtCi * y_local;
    
    arma::mat inv_chol_prec = arma::inv(arma::trimatl( arma::chol(XtCiX, "lower") ));
    
    arma::vec sub_rand_mvn = arma::trans( rand_mvn.submat(j, 0, j, j-1) );
    arma::vec b_post = inv_chol_prec.t() * ( inv_chol_prec * XtCiy + sub_rand_mvn);
    
    Bcond.submat(j, 0, j, j-1) = arma::trans( b_post );
  }
  
  arma::mat Iq = arma::eye(q,q);
  A_ = arma::solve(arma::trimatl(Iq - Bcond), arma::diagmat(sigmas));

}
