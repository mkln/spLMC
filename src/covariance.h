
#ifndef XCOV_LMC 
#define XCOV_LMC

#include <RcppArmadillo.h>
#include "omp_import.h"

const int MAT_NU_MAX=3;

using namespace std;


// 0: power exponential
// 1: matern
// 2: wave

arma::mat Correlationf(const arma::mat& coords, const arma::uvec& ix, const arma::uvec& iy, 
                       const arma::vec& theta, double * bessel_ws, int matern, bool same);

arma::mat Correlationc(const arma::mat& coordsx, const arma::mat& coordsy, 
                       const arma::vec& theta, int matern, bool same);

#endif
