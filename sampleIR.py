
def sampleIR(K, q, R, M, thetas):
    N = len(R)  # Number of possible observations
    Theta_a = [np.argmax([np.sum(q[theta][a_prime] * R) for a_prime in range(K)]) for theta in thetas]
    p_a_star = np.array([Theta_a.count(a_star) / M for a_star in range(K)])

    p_a_y = np.array([[np.mean([q[theta][a][y] for theta in thetas]) for y in range(N)] for a in range(K)])
    p_a_a_star_y = np.array([[[np.mean([q[theta][a][y] for theta in thetas if Theta_a[theta] == a_star]) for y in
                               range(N)] for a_star in range(K)] for a in range(K)])

    R_star = np.sum(p_a_y * R, axis=1)
    g_a = np.array([-np.sum(
        p_a_a_star_y[a][a_star] * np.log((p_a_a_star_y[a][a_star] / (p_a_star[a_star] * p_a_y[a] + 1e-10)) + 1e-10)) for
                    a_star in range(K) for a in range(K)])
    Delta_a = R_star - np.array([np.mean([np.sum(q[theta][a] * R) for theta in thetas]) for a in range(K)])

    return Delta_a, g_a


# Example usage
K, M = 5, 100  # Number of actions, samples
R = np.random.rand(10)  # Reward function over 10 possible observations
q = np.random.rand(M, K, 10)  # Transition probabilities
thetas = np.random.randint(0, M, size=M)  # Sampled thetas

Delta, g = sampleIR(K, q, R, M, thetas)
print("Delta:", Delta)
print("g:", g)
