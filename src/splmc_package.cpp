#include "lmc.h"
#include "interrupt.h"


using namespace std;

// [[Rcpp::export]]
Rcpp::List lmc_response(const arma::mat& Y, 
                          const arma::mat& coords,
                          
                          const arma::field<arma::uvec>& custom_dag,
                          
                          arma::mat theta_opts, 
                          const arma::mat& A_start,
                          
                          int mcmc = 1000,
                          int print_every = 100,
                          int dag_opts = 0,
                          bool upd_A = true,
                          bool upd_theta = true,
                          int num_threads = 1){
  
  Rcpp::Rcout << "GP-LMC response model." << endl;
  
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#else
  if(num_threads > 1){
    Rcpp::warning("num_threads > 1, but source not compiled with OpenMP support.");
    num_threads = 1;
  }
#endif
  
  unsigned int q = Y.n_cols;
  unsigned int n = Y.n_rows;
  
  if(print_every > 0){
    Rcpp::Rcout << "Preparing..." << endl;
  }
  
  LMC lmc(Y, coords, A_start, theta_opts, custom_dag, dag_opts,
                  num_threads);
  
  // storage
  arma::cube theta = arma::zeros(4, q, mcmc);
  arma::cube Sigma = arma::zeros(q, q, mcmc);
  
  if(print_every > 0){
    Rcpp::Rcout << "Starting MCMC" << endl;
  }
  
  for(unsigned int m=0; m<mcmc; m++){
    
    if(upd_theta){
      lmc.upd_thetaj_metrop();
    }
    if(upd_A){
      lmc.upd_A_metrop();
    }
    
    Sigma.slice(m) = lmc.A_ * lmc.A_.t();
    theta.slice(m) = lmc.theta_options;
    
    bool print_condition = (print_every>0);
    if(print_condition){
      print_condition = print_condition & (!(m % print_every));
    };
    if(print_condition){
      Rcpp::Rcout << "Iteration: " <<  m+1 << " of " << mcmc << endl;
    }
    
    bool interrupted = checkInterrupt();
    if(interrupted){
      Rcpp::stop("Interrupted by the user.");
    }
  }
  
  return Rcpp::List::create(
    Rcpp::Named("Sigma") = Sigma,
    Rcpp::Named("theta") = theta,
    Rcpp::Named("dag_cache") = lmc.daggp_options[0].dag_cache
  );
  
}
