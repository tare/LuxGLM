data {
  int<lower=1> K; // number of cytosines

  int<lower=1> P; // number of predictors

  int<lower=1> N; // number of samples
	
  int<lower=1> M; // number of libraries

  real<lower=0,upper=1> bsEff[M];
  real<lower=0,upper=1> seqErr[M];
  real bsBEff;

  int<lower=0> bsC[K,M];
  int<lower=0> bsTot[K,M];

  matrix[N,P] D; // design matrix
  int tr2br[M]; // grouping

  vector[P] mu_B;
  cov_matrix[P] V_B_U_B;

  real<lower=0> alpha;
  real<lower=0> beta;
  cov_matrix[N] V_E_U_E;

}

transformed data {

  cholesky_factor_cov[P] chol_V_B_U_B;

  chol_V_B_U_B = cholesky_decompose(V_B_U_B);
}

parameters {
  vector[P] B[K];

  real<lower=0> sigma2_E[K];

  vector[N] Y[K];

}

transformed parameters {
  real<lower=0,upper=1> theta[K,N];

  for (s in 1:N) {
    for (n in 1:K) {
      theta[n,s] = inv_logit(Y[n,s]');
    }
  }
}

model {
  for (n in 1:K) {
    sigma2_E[n] ~ gamma(alpha,beta);
    B[n] ~ multi_normal_cholesky(mu_B,chol_V_B_U_B);
    Y[n] ~ multi_normal_prec(D*B[n],sigma2_E[n]*V_E_U_E);
  }

  for (s in 1:M) {
    for (n in 1:K) {
      bsC[n,s] ~ binomial(bsTot[n,s],
        (1-theta[n,tr2br[s]])*((1.0 - seqErr[s])*(1.0 - bsEff[s]) + seqErr[s] * bsEff[s]) +
        theta[n,tr2br[s]]*((1.0 - bsBEff)*(1.0 - seqErr[s]) + seqErr[s] * bsBEff));
    }
  }
}



