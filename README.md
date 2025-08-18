# GP-LMC Response Models

This package implements Gaussian Process Linear Model of Coregionalization (GP-LMC) response models with efficient computation based on Vecchia and DAG-based GP approximations.

The main entry point is the function `lmc_response()`, which runs an adaptive Metropolis MCMC sampler for the latent GP hyperparameters and the factor loadings matrix.

Find an example in `example/example.r`