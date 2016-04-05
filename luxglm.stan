data {
  real mu_mu_bsEff;
  real sigma_mu_bsEff;
  
  real mu_sigma_bsEff;
  real sigma_sigma_bsEff;

  real mu_mu_bsBEff;
  real sigma_mu_bsBEff;

  real mu_sigma_bsBEff;
  real sigma_sigma_bsBEff;

  real mu_mu_oxEff;
  real sigma_mu_oxEff;

  real mu_sigma_oxEff;
  real sigma_sigma_oxEff;

  real mu_mu_seqErr;
  real sigma_mu_seqErr;

  real mu_sigma_seqErr;
  real sigma_sigma_seqErr;

  int<lower=1> K; // number of cytosines

  int<lower=1> P; // number of predictors

  int<lower=1> N; // number of rows

  int<lower=0> bsC[K,N];
  int<lower=0> bsTot[K,N];
  int<lower=0> oxC[K,N];
  int<lower=0> oxTot[K,N];

  matrix[N,P] D; // design matrix
  matrix[3*N,3*P] I_D;

  matrix[P,3] mu_B;
  cov_matrix[P*3] V_B_U_B;

  real<lower=0> alpha;
  real<lower=0> beta;
  cov_matrix[N*3] V_E_U_E;

  int<lower=1> K_control; // number of control cytosines

  int<lower=0> bsC_control[K_control,N];
  int<lower=0> bsTot_control[K_control,N];
  int<lower=0> oxC_control[K_control,N];
  int<lower=0> oxTot_control[K_control,N];

  vector<lower=0>[3] alpha_control[K_control]; // prior for control cytosines
}

transformed data {
  vector[P*3] vec_mu_B;

  cholesky_factor_cov[P*3] chol_V_B_U_B;

  for (j in 1:3) {
    for (i in 1:P) {
      vec_mu_B[(j-1)*P+i] <- mu_B[i,j];
    }
  }

  chol_V_B_U_B <- cholesky_decompose(V_B_U_B);
}

parameters {
  real mu_bsEff;
  real<lower=0> sigma_bsEff;
  real raw_bsEff[N];

  real mu_bsBEff;
  real<lower=0> sigma_bsBEff;
  real raw_bsBEff[N];

  real mu_oxEff;
  real<lower=0> sigma_oxEff;
  real raw_oxEff[N];

  real mu_seqErr;
  real<lower=0> sigma_seqErr;
  real raw_seqErr[N];

  matrix[P,3] B[K];

  real<lower=0> sigma2_E[K];

  matrix[N,3] Y[K];

  simplex[3] theta_control[K_control,N];
}

transformed parameters {
  real<lower=0,upper=1> bsEff[N];
  real<lower=0,upper=1> bsBEff[N];
  real<lower=0,upper=1> oxEff[N];
  real<lower=0,upper=1> seqErr[N];

  simplex[3] theta[K,N];

  for (s in 1:N) {
    bsEff[s]  <- inv_logit(mu_bsEff  + sigma_bsEff  * raw_bsEff[s]);
    bsBEff[s] <- inv_logit(mu_bsBEff + sigma_bsBEff * raw_bsBEff[s]);
    oxEff[s]  <- inv_logit(mu_oxEff  + sigma_oxEff  * raw_oxEff[s]);
    seqErr[s] <- inv_logit(mu_seqErr + sigma_seqErr * raw_seqErr[s]);

    for (n in 1:K) {
      theta[n,s] <- softmax(Y[n,s]');
    }
  }
}

model {
  for (n in 1:K) {
    sigma2_E[n] ~ gamma(alpha,beta);
    to_vector(B[n]) ~ multi_normal_cholesky(vec_mu_B,chol_V_B_U_B);
    to_vector(Y[n]) ~ multi_normal_prec(I_D*to_vector(B[n]),sigma2_E[n]*V_E_U_E);
  }

  for (s in 1:N) {
    for (n in 1:K) {
      bsC[n,s] ~ binomial(bsTot[n,s],
        theta[n,s,1]*((1.0 - seqErr[s])*(1.0 - bsEff[s]) + seqErr[s] * bsEff[s]) +
        theta[n,s,2]*((1.0 - bsBEff[s])*(1.0 - seqErr[s]) + seqErr[s] * bsBEff[s]) +
        theta[n,s,3]*((1.0 - bsBEff[s])*(1.0 - seqErr[s]) + seqErr[s] * bsBEff[s]));
      oxC[n,s] ~ binomial(oxTot[n,s],
        theta[n,s,1]*((1.0 - seqErr[s])*(1.0 - bsEff[s]) + seqErr[s] * bsEff[s]) +
        theta[n,s,2]*((1.0 - bsBEff[s])*(1.0 - seqErr[s]) + seqErr[s] * bsBEff[s]) +
        theta[n,s,3]*(oxEff[s]*((1.0 - seqErr[s])*(1.0 - bsEff[s]) + seqErr[s] * bsEff[s])+(1.0 - oxEff[s])*((1.0 - bsBEff[s])*(1.0 - seqErr[s]) + bsBEff[s] * seqErr[s])));
    }

    for (n in 1:K_control) {
      theta_control[n,s] ~ dirichlet(alpha_control[n]);
      bsC_control[n,s] ~ binomial(bsTot_control[n,s],
        theta_control[n,s,1]*((1.0 - seqErr[s])*(1.0 - bsEff[s]) + seqErr[s] * bsEff[s]) +
        theta_control[n,s,2]*((1.0 - bsBEff[s])*(1.0 - seqErr[s]) + seqErr[s] * bsBEff[s]) +
        theta_control[n,s,3]*((1.0 - bsBEff[s])*(1.0 - seqErr[s]) + seqErr[s] * bsBEff[s]));
      oxC_control[n,s] ~ binomial(oxTot_control[n,s],
        theta_control[n,s,1]*((1.0 - seqErr[s])*(1.0 - bsEff[s]) + seqErr[s] * bsEff[s]) +
        theta_control[n,s,2]*((1.0 - bsBEff[s])*(1.0 - seqErr[s]) + seqErr[s] * bsBEff[s]) +
        theta_control[n,s,3]*(oxEff[s]*((1.0 - seqErr[s])*(1.0 - bsEff[s]) + seqErr[s] * bsEff[s])+(1.0 - oxEff[s])*((1.0 - bsBEff[s])*(1.0 - seqErr[s]) + bsBEff[s] * seqErr[s])));
    }
  }

  mu_bsEff ~ normal(mu_mu_bsEff, sigma_mu_bsEff);
  sigma_bsEff ~ lognormal(mu_sigma_bsEff, sigma_sigma_bsEff);
  raw_bsEff ~ normal(0,1);

  mu_bsBEff ~ normal(mu_mu_bsBEff, sigma_mu_bsBEff);
  sigma_bsBEff ~ lognormal(mu_sigma_bsBEff, sigma_sigma_bsBEff);
  raw_bsBEff ~ normal(0,1);

  mu_oxEff ~ normal(mu_mu_oxEff, sigma_mu_oxEff);
  sigma_oxEff ~ lognormal(mu_sigma_oxEff, sigma_sigma_oxEff);
  raw_oxEff ~ normal(0,1);

  mu_seqErr ~ normal(mu_mu_seqErr, sigma_mu_seqErr);
  sigma_seqErr ~ lognormal(mu_sigma_seqErr, sigma_sigma_seqErr);
  raw_seqErr ~ normal(0,1);
}
