data {
  int<lower=1> K; // number of cytosines

  int<lower=1> P; // number of predictors

  int<lower=1> N; // number of rows
	
  int<lower=1> M; // number of rows

  real<lower=0,upper=1> bsEff[M];
  real<lower=0,upper=1> seqErr[M];
  real bsBEff;

  int<lower=0> bsC[K,M];
  int<lower=0> bsTot[K,M];

  matrix[N,P] D; // design matrix
  int tr2br[M]; // design matrix
  matrix[N,P] I_D;

  matrix[P,1] mu_B;
  cov_matrix[P] V_B_U_B;

  real<lower=0> alpha;
  real<lower=0> beta;
  cov_matrix[N] V_E_U_E;

}

transformed data {
  vector[P] vec_mu_B;

  cholesky_factor_cov[P] chol_V_B_U_B;

  for (i in 1:P) {
    vec_mu_B[i] = mu_B[i,1];
  }

  chol_V_B_U_B = cholesky_decompose(V_B_U_B);
}

parameters {
  matrix[P,1] B[K];

  real<lower=0> sigma2_E[K];

  matrix[N,1] Y[K];

}

transformed parameters {
  real<lower=0,upper=1> theta[K,N];

  for (s in 1:N) {
    for (n in 1:K) {
      theta[n,s] = inv_logit(Y[n,s,1]');
    }
  }
}

model {
  for (n in 1:K) {
    sigma2_E[n] ~ gamma(alpha,beta);
    to_vector(B[n]) ~ multi_normal_cholesky(vec_mu_B,chol_V_B_U_B);
    to_vector(Y[n]) ~ multi_normal_prec(I_D*to_vector(B[n]),sigma2_E[n]*V_E_U_E);
  }

  for (s in 1:M) {
    for (n in 1:K) {
      bsC[n,s] ~ binomial(bsTot[n,s],
        (1-theta[n,tr2br[s]])*((1.0 - seqErr[s])*(1.0 - bsEff[s]) + seqErr[s] * bsEff[s]) +
        theta[n,tr2br[s]]*((1.0 - bsBEff)*(1.0 - seqErr[s]) + seqErr[s] * bsBEff));
    }
  }
}



