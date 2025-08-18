# GP-LMC Response Models

This package implements Gaussian Process Linear Model of Coregionalization (GP-LMC) response models with efficient computation based on Vecchia and DAG-based GP approximations.

The main entry point is the function `lmc_response()`, which runs an adaptive Metropolis MCMC sampler for the latent GP hyperparameters and the factor loadings matrix.

Find an example in `example/example.r`

## Math details

We observe a data matrix \(Y \in \mathbb{R}^{n \times q}\), where \(n\) is the number of spatial locations and \(q\) is the number of outcomes.  
Let \(\mathrm{vec}(Y)\) denote the column-stacked vectorization of \(Y\).

We assume

\[
\mathrm{vec}(Y) \;\sim\; \mathcal{N}\!\left( 
  0, \;
  (A \otimes I_n) 
  \, \mathrm{blkdiag}\!\big(R_1, \ldots, R_q\big) \,
  (A^\top \otimes I_n)
\right),
\]

where  

- \(A \in \mathbb{R}^{q \times q}\) is the factor loadings (coregionalization) matrix,  
- \(I_n\) is the \(n \times n\) identity matrix,  
- \(R_j \in \mathbb{R}^{n \times n}\) is the correlation matrix of the \(j\)-th latent Gaussian process (Matern), 
- \(\mathrm{blkdiag}(R_1, \ldots, R_q)\) denotes the block-diagonal matrix with \(R_j\) blocks.  

This package **does not** fit models with measurement error and **does not** allow to set the number of factors to less than the number of outcomes and **does not** allow the estimation of linear covariate effects. For all these features, consider R packages `meshed` and `spBayes`.
