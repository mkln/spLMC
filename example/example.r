library(tidyverse)
library(spLMC)
# spiox package required to run dag_vecchia() 
# remotes::install_github("mkln/spiox")

set.seed(25) 
n_threads <- 16

q <- 6
# spatial coordinates
coords <- expand.grid(x <- seq(0, 1, length.out=20), x) %>%
  as.matrix()
colnames(coords) <- c("Var1","Var2")
nr <- nrow(coords)

iox <- FALSE

# lmc data
Sigma <- solve(rWishart(1, q+1, diag(q))[,,1])
A <- t(chol(Sigma))
custom_dag <- spiox::dag_vecchia(coords, 15, TRUE)

phis <- c(30, 10, 1, 50) %>% sample(q, replace=T) 

# sample LMC data
U <- rnorm(nr * q) %>% matrix(ncol=q)
Llist <- 1:q %>% lapply(\(j) t(chol( exp(- phis[j] * as.matrix(dist(coords))) )) )
V <- 1:q %>% lapply(\(j) Llist[[j]] %*% U[,j]) %>% abind::abind(along=2)
Y <- V %*% t(A)


m_nn <- 15
mcmc <- 2000

# import this function from spiox package
if (!requireNamespace("spiox", quietly = TRUE)) remotes::install_github("mkln/spiox")
custom_dag <- spiox::dag_vecchia(coords, m_nn, TRUE)

# this uses the same interface as spiox
# fix everything but the first row here to do so
theta_opts <- rbind(c(10,15,1,1,1,1,1)[1:q], 
                    rep(1, q), 
                    rep(0.5, q), 
                    rep(1e-8, q)) 

system.time({
  splmc_out <- lmc_response(Y, coords, custom_dag, theta_opts, diag(q),
                                   mcmc = mcmc, 
                                   print_every=100, 
                                   dag_opts= -1, # for gridded data
                                   upd_A = TRUE, 
                                   upd_theta = TRUE, 
                                  n_threads) })

plot_lmc <- function(lmc_out, perc_show=0.75){
  q     <- dim(lmc_out$Sigma)[1]
  mcmc  <- dim(lmc_out$Sigma)[3]
  idx   <- (floor((1-perc_show) * mcmc) + 1):mcmc  # last 3/4
  
  # diagonals across iterations
  sigma_mat <- sapply(1:q, function(j) lmc_out$Sigma[j, j, idx])
  phi_mat <- sapply(1:q, function(j) lmc_out$theta[1, j, idx])
  
  df_sigma <- as.data.frame(sigma_mat) |>
    mutate(iter = idx) |>
    pivot_longer(-iter, names_to = "j", values_to = "value") |>
    mutate(param = "sigma2", j = as.integer(gsub("\\D", "", j)))
  
  df_phi <- as.data.frame(phi_mat) |>
    mutate(iter = idx) |>
    pivot_longer(-iter, names_to = "j", values_to = "value") |>
    mutate(param = "phi", j = as.integer(gsub("\\D", "", j)))
  
  draw <- bind_rows(df_sigma, df_phi)
  
  plotted <- ggplot(draw, aes(iter, value)) +
    geom_line(linewidth = 0.3) +
    facet_wrap(param ~ j, scales = "free_y", ncol=q) +
    labs(x = "Iteration", y = "Value") +
    theme_minimal()
  
  return(plotted)
}

splmc_out %>% plot_lmc(0.5)

# zero distance correlation
Omega_lmc <- splmc_out$Sigma %>% apply(3, \(s) cov2cor(s)) %>% array(dim=c(q,q,mcmc))

Omega_lmc[2,3,] %>% plot(type='l')
