#include <RcppArmadillo.h>

// log multivariate gamma: log Γ_q(a) = (q(q-1)/4) log π + Σ_{j=1}^q log Γ(a + (1-j)/2)
inline double log_multigamma(double a, int q){
  const double c = 0.25 * q * (q - 1) * std::log(M_PI);
  double s = 0.0;
  for(int j=1; j<=q; ++j) s += std::lgamma(a + 0.5*(1 - j));
  return c + s;
}


inline double log_iwish_density(const arma::mat& A, const arma::mat& S, int v){
  const int q = A.n_rows;
  if (A.n_cols != (unsigned)q || S.n_rows != (unsigned)q || S.n_cols != (unsigned)q)
    Rcpp::stop("A and S must be square q x q with the same q.");
  if (v <= q - 1) return -INFINITY;  // df condition
  
  // SPD checks via Cholesky
  arma::mat LA, LS;
  if(!arma::chol(LA, A, "lower")) return -INFINITY;
  if(!arma::chol(LS, S, "lower")) return -INFINITY;
  
  // log-determinants
  const double logdetA = 2.0 * arma::sum(arma::log(LA.diag()));
  const double logdetS = 2.0 * arma::sum(arma::log(LS.diag()));
  
  // trace(S A^{-1}) computed via solve: X = A^{-1} S  => tr(S A^{-1}) = tr(X)
  arma::mat X = arma::solve(A, S);  // A X = S
  const double trSAinv = arma::trace(X);
  
  // assemble log-density
  const double v2 = 0.5 * v;
  const double term_const = v2 * logdetS
  - (v * q * 0.5) * std::log(2.0)
    - log_multigamma(v2, q);
    const double term_A = -0.5 * ( (v + q + 1.0) * logdetA + trSAinv );
    
    return term_const + term_A;
}