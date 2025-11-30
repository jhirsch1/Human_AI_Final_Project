data {
    int<lower=1> N;                          // number of transitions
    array[N] real thetaM_t;                  // observed trust at time t
    array[N] real thetaM_tp1;                // observed trust at time t+1

    int<lower=1> E;                          // number of event types
    array[N] int<lower=1, upper=E> e_id;     // event_type index per transition
}

parameters {
    array[N] real theta_t;              // latent trust at time t
    array[N] real theta_tp1;            // latent trust at time t+1

    vector[E] alpha;                    // trust dynamics slope per event e
    vector[E] beta;                     // trust dynamics intercept per event e
    vector<lower=0>[E] sigma_e;         // process noise per event e

    real<lower=0> sigma_obs;            // observation noise for Muir scores
}

model {
    // ---------- Priors ----------
    alpha ~ normal(0, 1);
    beta  ~ normal(0, 3);
    sigma_e ~ exponential(1);
    sigma_obs ~ exponential(1);

    // ---------- Likelihood ----------
    for (n in 1:N) {
        int e = e_id[n];
        // True latent trust follows linear gaussian dynamics
        theta_tp1[n] ~ normal(alpha[e] * theta_t[n] + beta[e],
                              sigma_e[e]);

        // Observed Muir trust is noisy measurement
        thetaM_t[n]   ~ normal(theta_t[n], sigma_obs);
        thetaM_tp1[n] ~ normal(theta_tp1[n], sigma_obs);
    }
}