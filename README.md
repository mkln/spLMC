# GP-LMC Response Models

This package implements Gaussian Process Linear Model of Coregionalization (GP-LMC) response models with efficient computation based on Vecchia and DAG-based GP approximations.

The main entry point is the function `lmc_response()`, which runs an adaptive Metropolis MCMC sampler for the latent GP hyperparameters and the factor loadings matrix.

Find an example in `example/example.r`

## Math details

We observe a data matrix ![Y](https://latex.codecogs.com/svg.latex?Y\in\mathbb{R}^{n\times%20q}), where ![n](https://latex.codecogs.com/svg.latex?n) is the number of spatial locations and ![q](https://latex.codecogs.com/svg.latex?q) is the number of outcomes.  
Let ![vec(Y)](https://latex.codecogs.com/svg.latex?\mathrm{vec}(Y)) denote the column-stacked vectorization of ![Y](https://latex.codecogs.com/svg.latex?Y).

We assume

![likelihood](https://latex.codecogs.com/svg.latex?\mathrm{vec}(Y)\sim\mathcal{N}\!\left(0,\;(A\otimes%20I_n)\,\mathrm{blkdiag}(R_1,\ldots,R_q)\,(A^\top\otimes%20I_n)\right))

where  

- ![A](https://latex.codecogs.com/svg.latex?A\in\mathbb{R}^{q\times%20q}) is the factor loadings (coregionalization) matrix,  
- ![I_n](https://latex.codecogs.com/svg.latex?I_n) is the ![n\times n](https://latex.codecogs.com/svg.latex?n\times%20n) identity matrix,  
- ![R_j](https://latex.codecogs.com/svg.latex?R_j\in\mathbb{R}^{n\times%20n}) is the correlation matrix of the ![j](https://latex.codecogs.com/svg.latex?j)-th latent Gaussian process (Matérn),  
- ![blkdiag](https://latex.codecogs.com/svg.latex?\mathrm{blkdiag}(R_1,\ldots,R_q)) denotes the block-diagonal matrix with ![R_j](https://latex.codecogs.com/svg.latex?R_j) blocks.  

---

This package **does not** fit models with measurement error, **does not** allow setting the number of factors to less than the number of outcomes, and **does not** estimate linear covariate effects.  

For these additional features, consider the R packages [`meshed`](https://github.com/mkln/meshed) and [`spBayes`](https://cran.r-project.org/package=spBayes).