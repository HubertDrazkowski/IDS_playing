import numpy as np
from scipy.stats import norm


def independentGaussianVIR(K, mu, sigma):
    # Step 1-2: Define Gaussian PDF and CDF for each action
    fa = lambda x, a: norm.pdf(x, mu[a], sigma[a])
    Fa = lambda x, a: norm.cdf(x, mu[a], sigma[a])

    # Step 3: Calculate the joint CDF F(x)
    F = lambda x: np.prod([Fa(x, a) for a in range(K)])

    # Step 4-5: Compute p*(a) and Ma|a using numerical integration
    p_star = np.zeros(K)
    Ma_a = np.zeros(K)
    for a in range(K):
        p_star[a] = np.trapz([fa(x, a) / Fa(x, a) * F(x) for x in np.linspace(-10, 10, 1000)], dx=0.01)
        Ma_a[a] = 1 / p_star[a] * np.trapz([x * fa(x, a) / Fa(x, a) * F(x) for x in np.linspace(-10, 10, 1000)],
                                           dx=0.01)

    # Step 6: Calculate Ma'|a for a ≠ a'
    Ma_prime_a = np.zeros((K, K))
    for a in range(K):
        for a_prime in range(K):
            if a != a_prime:
                Ma_prime_a[a, a_prime] = mu[a_prime] - sigma[a_prime] ** 2 / p_star[a] * np.trapz(
                    [fa(x, a) * fa(x, a_prime) / (Fa(x, a) * Fa(x, a_prime)) * F(x) for x in
                     np.linspace(-10, 10, 1000)],
                    dx=0.01
                )

    # Step 7-9: Calculate ρ*, ∆a, and va
    rho_star = np.sum(p_star * Ma_a)
    Delta_a = rho_star - mu
    va = np.array([np.sum(p_star[a_prime] * (Ma_a[a_prime] - mu[a]) ** 2 for a_prime in range(K)) for a in range(K)])

    # Step 10: Return the computed measures
    return Delta_a, va


# Example usage
K = 3  # Number of arms
mu = np.array([1.0, 2.0, 3.0])  # Current beliefs about the means
sigma = np.array([0.5, 0.5, 0.5])  # Current beliefs about the standard deviations

Delta, v = independentGaussianVIR(K, mu, sigma)
print("Delta:", Delta)
print("v:", v)
