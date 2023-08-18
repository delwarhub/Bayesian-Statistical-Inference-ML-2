data {
  int<lower = 1> N;
  vector[N] ncond;
  vector[N] RT;
  int nresp[N];
}
parameters {
  real alpha[2];
  real beta[2];
  real<lower = 0> sigma;
  real<lower = 0, upper = min(RT)> T_nd;
}
model {
  vector[N] T = RT - T_nd;
  target += normal_lpdf(alpha | 7, 1);
  target += normal_lpdf(beta | 0, .2);
  target += normal_lpdf(sigma | .5, .2)
    - normal_lccdf(0 | .5, .2);
    target += normal_lpdf(T_nd | 150, 100)
    - log_diff_exp(normal_lcdf(min(RT) | 150, 100),
                   normal_lcdf(0 | 150, 100));
  for(n in 1:N){
    if(nresp[n] == 1) // grammatical
      target += lognormal_lpdf(T[n] | alpha[1] -
                               ncond[n] * beta[1], sigma)  +
        lognormal_lccdf(T[n] | alpha[2] -
                        ncond[n] * beta[2] , sigma);
    else
       target += lognormal_lpdf(T[n] | alpha[2] -
                                ncond[n] * beta[2], sigma) +
        lognormal_lccdf(T[n] | alpha[1] -
                        ncond[n] * beta[1], sigma);
  }
}

