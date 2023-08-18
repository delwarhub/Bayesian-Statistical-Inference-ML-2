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
  real<lower = 0.1, upper = 1> theta;
  real<lower = 0.1, upper = 1> p1;
  real<lower = 0.1, upper = p1> p2;
}
model {
  // priors for the task component
  target += normal_lpdf(mu | 5.5, 0.5)
    - normal_lccdf(5.2 | 5.5, .2);
  target += normal_lpdf(sigma | .1, .05)
    - normal_lccdf(0 | .1, .05);
  target += normal_lpdf(delta | .2, .05)
    - normal_lccdf(0 | .2, .05);
  target += normal_lpdf(theta | 0, 0.25);
  target += beta_lpdf(p1 | 8, 2);
  target += beta_lpdf(p2 | 2, 8);
  for(n in 1:N) {
    if(nresp[n] == 1){
      target += log_sum_exp(log(theta) +
                          lognormal_lpdf(RT[n] | mu + ncond[n] * delta, sigma) +
                          bernoulli_lpmf(nresp[n] | p1),
                          log1m(theta) +
                          lognormal_lpdf(RT[n] | mu, sigma) +
                          bernoulli_lpmf(nresp[n] | p2));
    }
    else{
      target += lognormal_lpdf(RT[n] | mu, sigma) +
                          bernoulli_lpmf(nresp[n] | p2) ;
    }
    
}
}
