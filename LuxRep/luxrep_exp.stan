data {
  real mu_mu_bsEff;
  real sigma_mu_bsEff;
  
  real mu_sigma_bsEff;
  real sigma_sigma_bsEff;

  real mu_mu_seqErr;
  real sigma_mu_seqErr;

  real mu_sigma_seqErr;
  real sigma_sigma_seqErr;

  real bsBEff_fixed;

  int<lower=1> N; // number of rows
	
  int<lower=1> K_control; // number of control cytosines

  int<lower=0> bsC_control[K_control];
  int<lower=0> bsTot_control[K_control];

  vector<lower=0>[2] alpha_control[K_control]; // prior for control cytosines
}

parameters {
  real mu_bsEff;
  real<lower=0> sigma_bsEff;
  real raw_bsEff[N];

  real mu_seqErr;
  real<lower=0> sigma_seqErr;
  real raw_seqErr[N];

  simplex[2] theta_control[K_control];
}

transformed parameters {
  real<lower=0,upper=1> bsEff[N];
  real<lower=0,upper=1> seqErr[N];

  for (s in 1:N) {
    bsEff[s]  = inv_logit(mu_bsEff  + sigma_bsEff  * raw_bsEff[s]);
    seqErr[s] = inv_logit(mu_seqErr + sigma_seqErr * raw_seqErr[s]);
  }
}

model {
  for (s in 1:N) {
    for (n in 1:K_control) {
      theta_control[n] ~ dirichlet(alpha_control[n]);
      bsC_control[n] ~ binomial(bsTot_control[n],
        theta_control[n,1]*((1.0 - seqErr[s])*(1.0 - bsEff[s]) + seqErr[s] * bsEff[s]) +
        theta_control[n,2]*((1.0 - bsBEff_fixed)*(1.0 - seqErr[s]) + seqErr[s] * bsBEff_fixed));
    }
  }

  mu_bsEff ~ normal(mu_mu_bsEff, sigma_mu_bsEff);
  sigma_bsEff ~ lognormal(mu_sigma_bsEff, sigma_sigma_bsEff);
  raw_bsEff ~ normal(0,1);

  mu_seqErr ~ normal(mu_mu_seqErr, sigma_mu_seqErr);
  sigma_seqErr ~ lognormal(mu_sigma_seqErr, sigma_sigma_seqErr);
  raw_seqErr ~ normal(0,1);
}


