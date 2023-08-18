data {
  int<lower=1> N_obs;
  real RT[N_obs];
  int<lower=0,upper=1> n_resp[N_obs];
  int<lower=0,upper=1> n_cond[N_obs];
}
parameters {
  real<lower=0.1,upper=1> theta_guess;
  real<lower=0.1,upper=1> theta_retr;
  real mu_1; // time to decide guess or not
  real mu_3; // time to retrieve if condition attractor
  real<upper=mu_3> mu_4; // time to retrieve if baseline condition
  real <upper=mu_4> mu_2; // time to guess
  
  real<lower = 0> sigma;
}
transformed parameters {
  simplex[2] theta[N_obs];
  simplex[2] theta_G[N_obs];
  simplex[2] theta_UG[N_obs];
  for(n in 1:N_obs){
    //Pr_Grammatical:
    theta[n, 1] = theta_guess*0.5 + (1-theta_guess)*theta_retr;
    //Pr_Ungrammatical:
    theta[n, 2] = theta_guess*0.5 + (1-theta_guess)*(1-theta_retr);
    //Pr_guess_when_grammatical
    theta_G[n, 1] = (theta_guess*0.5)/theta[n, 1];
    //Pr_retrieval_when_grammatical
    theta_G[n, 2] = ((1-theta_guess)*theta_retr)/theta[n, 1];
    //Pr_guess_when_ungrammatical
    theta_UG[n, 1] = (theta_guess*0.5)/theta[n, 2];
    //Pr_retrieval_when_ungrammatical
    theta_UG[n, 2] = ((1-theta_guess)*(1-theta_retr))/theta[n, 2];
  }
}
model {
  target += beta_lpdf(theta_guess | 2, 2);
  target += beta_lpdf(theta_retr | 2, 2);
  target += normal_lpdf(mu_1 | 7, 1);
  target += normal_lpdf(mu_3 | 4, 1);
  target += normal_lpdf(mu_4 | 3, 1) 
    - normal_lcdf(mu_3 | 4, 1);
  target += normal_lpdf(mu_2 | 2, 1) 
    - normal_lcdf(mu_4 | 3, 1);
  
  target += normal_lpdf(sigma | .1, .05)
    - normal_lccdf(0 | .1, .05);
  for(n in 1:N_obs)
  {
    if (n_cond[n] == 0) // no attraction effect
    {
         if (n_resp[n]==0) // if response is Ungrammatical
      {
        target += log(theta[n,2]) + log_sum_exp(log(theta_UG[n, 1])+
        lognormal_lpdf(RT[n] | mu_1 + mu_2, sigma), log(theta_UG[n, 2])+
        lognormal_lpdf(RT[n] | mu_1 + mu_4, sigma)) ;
      }
      else {
        target += log(theta[n,1]) + log_sum_exp(log(theta_G[n, 1])+
        lognormal_lpdf(RT[n] | mu_1 + mu_2, sigma), log(theta_G[n, 2])+
        lognormal_lpdf(RT[n] | mu_1 + mu_4, sigma)) ;
      }
    }
    else // attraction effect
    {
        if (n_resp[n]==0) // if response is Ungrammatical
      {
        target += log(theta[n,2]) + log_sum_exp(log(theta_UG[n, 1])+
        lognormal_lpdf(RT[n] | mu_1 + mu_2, sigma), log(theta_UG[n, 2])+
        lognormal_lpdf(RT[n] | mu_1 + mu_3, sigma)) ;
      }
      else {
        target += log(theta[n,1]) + log_sum_exp(log(theta_G[n, 1])+
        lognormal_lpdf(RT[n] | mu_1 + mu_2, sigma), log(theta_G[n, 2])+
        lognormal_lpdf(RT[n] | mu_1 + mu_3, sigma)) ;
      }
    }
   
  }
}