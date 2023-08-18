data {
  int<lower = 1> N;
  vector[N] ncond;
  vector[N] RT;
  int nresp[N];
}
parameters {
  real<lower = 5.2> mu;
  real<lower = 0> delta;
  real<lower = 0> sigma;
  real<lower = 0.1, upper = 1> p2;
}
model {
  // priors for the task component
  target += normal_lpdf(mu | 5.5, 0.5)
    - normal_lccdf(5.2 | 5.5, .2);
  target += normal_lpdf(sigma | .1, .05)
    - normal_lccdf(0 | .1, .05);
  target += normal_lpdf(delta | .2, .05)
    - normal_lccdf(0 | .2, .05);
  target += beta_lpdf(p2 | 2, 8);
  for(n in 1:N) {
      target += lognormal_lpdf(RT[n] | mu, sigma) +
                          bernoulli_lpmf(nresp[n] | p2) ;
}
}
