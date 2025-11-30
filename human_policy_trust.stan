functions {
    real logistic_func(real x) {
        return 1.0 / (1.0 + exp(-x));
    }
}

data {
    int<lower=1> N;                          // number of datapoints
    array[N] real theta;                     // trust estimate (latent or observed)
    array[N] int<lower=0, upper=1> aH;       // 1 = stay, 0 = intervene

    int<lower=1> J;                          // number of object types
    array[N] int<lower=1, upper=J> obj_id;   // object type per datapoint

    vector[J] rS;                       // reward if robot succeeds
    vector[J] rF;                       // reward if robot fails
}

parameters {
    vector[J] gamma;        // slope: trust → robot success belief
    vector[J] eta;          // intercept
}

model {
    // ---------- Priors ----------
    gamma ~ normal(0, 1);
    eta   ~ normal(0, 1);

    // ---------- Likelihood ----------
    for (n in 1:N) {
        int j = obj_id[n];

        // Trust → belief robot will succeed on object j
        real b = logistic_func(gamma[j] * theta[n] + eta[j]);

        // Belief → probability human stays
        real p_stay = logistic_func(b * rS[j] + (1 - b) * rF[j]);

        aH[n] ~ bernoulli(p_stay);
    }
}