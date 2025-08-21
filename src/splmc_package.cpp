#include "lmc.h"
#include "interrupt.h"


using namespace std;

//' Fit a Gaussian Process Linear Model of Coregionalization (GP-LMC)
//'
//' Implements an adaptive Metropolis sampler for the GP-LMC response model.
//' The model assumes multivariate Gaussian process latent factors combined
//' through a factor-loading matrix \eqn{A}, with independent measurement error
//' across outcomes. The covariance of \eqn{Y} is
//' \deqn{ \mathrm{Cov}( \mathrm{vec}(Y) ) = (A \otimes I_n) \; \mathrm{blkdiag}\{R_j\} \; (A^\top \otimes I_n) }
//' where each \eqn{R_j} is the correlation matrix of the \eqn{j}-th latent process.
//'
//' @param Y \eqn{n \times q} matrix of outcomes, with \eqn{n} sites and \eqn{q} outcomes.
//' @param coords \eqn{n \times d} matrix of spatial coordinates for the \eqn{n} sites.
//' @param custom_dag A field of index vectors defining the Vecchia approximation
//'   DAG structure for each site. Use R package github.com/mkln/spiox for building the DAG
//' @param theta_opts A \eqn{4 \times q} matrix of latent GP hyperparameters, with rows
//'   corresponding to \eqn{(\phi, \sigma^2, \nu, \tau^2)} and columns to outcomes.
//' @param A_start Initial \eqn{q \times q} factor-loading matrix.
//' @param mcmc Integer, number of MCMC iterations (default 1000).
//' @param print_every Integer, print progress every this many iterations (default 100).
//' @param dag_opts Integer controlling Vecchia DAG construction for gridded data:
//'   \itemize{
//'     \item \code{-1}: build a cache assuming coords are gridded and the dag was built under the same assumption
//'     \item \code{0}: no change, use \code{custom_dag} as provided
//'     \item \code{>0}: prune the DAG with the given pruning parameter to facilitate ties in parent sets. Example: if number of neighbors was set to 20 and dag_opts=5 then each node in the dag is pruned by at most 5 edges to facilitate, resulting to final number of neighbors between 15 and 20. 
//'   }
//' @param upd_A Logical, whether to update the loading matrix \eqn{A} (default TRUE).
//' @param upd_theta Logical, whether to update GP hyperparameters \eqn{theta} (default TRUE).
//' @param num_threads Integer, number of OpenMP threads to use (default 1).
//'
//' @return A list with elements:
//' \item{Sigma}{\eqn{q \times q \times mcmc} array of posterior samples of \eqn{A A^\top}.}
//' \item{theta}{\eqn{4 \times q \times mcmc} array of posterior samples of hyperparameters.}
//' \item{dag_cache}{DAG structure used by the Vecchia approximation (for reference).}
//'
//' @details
//' This function constructs an \code{LMC} object internally and runs MCMC
//' updates for both the GP hyperparametersro and the factor loadings.
//' Storage is provided for posterior draws of the implied covariance
//' matrix \eqn{A A^\top} and the GP hyperparameters. Computation can be
//' parallelized using OpenMP if available.
//'
//' @examples
//' \dontrun{
//'   # Example data
//'   n <- 50; q <- 2
//'   coords <- matrix(runif(n*2), n, 2)
//'   Y <- matrix(rnorm(n*q), n, q)
//'   theta_opts <- matrix(1, 4, q)
//'   A_start <- diag(q)
//'   custom_dag <- some_dag_constructor(coords) # user-provided
//'
//'   fit <- lmc_response(Y, coords, custom_dag, theta_opts, A_start,
//'                       mcmc = 200, dag_opts = 0)
//'   str(fit)
//' }
//'
//' @export
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
                          int num_threads = 1,
                          bool matern = true,
                          bool debug = false){
  
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
                  num_threads, matern);
  
  // storage
  arma::cube theta = arma::zeros(4, q, mcmc);
  arma::cube Sigma = arma::zeros(q, q, mcmc);
  
  if(print_every > 0){
    Rcpp::Rcout << "Starting MCMC" << endl;
  }
  
  for(unsigned int m=0; m<mcmc; m++){
    
    if(upd_theta){
      try {
        lmc.upd_thetaj_metrop();
      } catch (std::exception &ex) {
        if(debug){
          Rcpp::Rcout << "Error in upd_thetaj_metrop at iteration " 
                      << m+1 << ": " << ex.what() << std::endl;
          
        }
        Rcpp::stop("MCMC stopped.\n");
      }
    }
    if(upd_A){
      try {
        lmc.upd_A_metrop();
      } catch (std::exception &ex) {
        if(debug){
          Rcpp::Rcout << "Error in upd_A_metrop at iteration " 
                      << m+1 << ": " << ex.what() << std::endl;
          
        }
        Rcpp::stop("MCMC stopped.\n");
      }
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

// [[Rcpp::export]]
double lmc_logdens(const arma::mat& Y, 
                        const arma::mat& coords,
                        
                        const arma::field<arma::uvec>& custom_dag,
                        
                        arma::mat theta, 
                        const arma::mat& Sigma,
                        
                        int dag_opts = 0,
                        int num_threads = 1,
                        bool matern = true){
  
  unsigned int q = Y.n_cols;
  unsigned int n = Y.n_rows;

  arma::mat A = arma::chol(Sigma, "lower");
  LMC lmc(Y, coords, A, theta, custom_dag, dag_opts,
          num_threads, matern);
  
  return lmc.logdens_curr_fast_();
  
}