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
  matrix[N*2,P*2] I_D;

  matrix[P,2] mu_B;
  cov_matrix[P*2] V_B_U_B;

  real<lower=0> alpha;
  real<lower=0> beta;
  cov_matrix[N*2] V_E_U_E;

}

transformed data {
  vector[P*2] vec_mu_B;

  cholesky_factor_cov[P*2] chol_V_B_U_B;

  for (j in 1:2) {
    for (i in 1:P) {
      vec_mu_B[(j-1)*P+i] = mu_B[i,j];
    }
  }

  chol_V_B_U_B = cholesky_decompose(V_B_U_B);
}

parameters {
  matrix[P,2] B[K];

  real<lower=0> sigma2_E[K];

  matrix[N,2] Y[K];

}

transformed parameters {
  simplex[2] theta[K,N];

  for (s in 1:N) {
    for (n in 1:K) {
      theta[n,s] = softmax(Y[n,s]');
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
        theta[n,tr2br[s],1]*((1.0 - seqErr[s])*(1.0 - bsEff[s]) + seqErr[s] * bsEff[s]) +
        theta[n,tr2br[s],2]*((1.0 - bsBEff)*(1.0 - seqErr[s]) + seqErr[s] * bsBEff));
    }
  }
}



