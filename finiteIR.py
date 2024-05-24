import numpy as np

# ================================
# Algorithms
# ================================
class finiteIR:
    """
    Replicates Algorithm 1 from Russo van Roy 2018 IDS
    Assuming R is suitable for vectorization, it is a little bit more fun then
    Args:
        L (int)
        K (int)
        N (int)
        R (function)
        p (L dim vector)
        q (L x K x N dim tensor)
    """
    def __init__(self, L, K, N, R, p, q):
        self.L = L  # Number of possible θ values
        self.K = K  # Number of actions
        self.N = N  # Number of possible observations
        self.R = R  # Reward function R(y)
        self.p = np.array(p)  # Prior probability mass function of theta
        self.q = np.array(q)  # Probability mass function of y given theta and a

    def update_posterior(self, pt, a, y):
        # Theta encapsulated in self.q[theta, a, y]
        # Update posterior distribution p_t(theta) using Bayes' rule
        numerator = pt * self.q[:, a, y]  # for all theta
        denominator = np.sum(numerator)  # sum over all theta
        if denominator == 0:
            return pt  # To handle division by zero
        else:
            return numerator / denominator


    def compute_Theta_a(self):
        Theta_a = {a: [] for a in range(self.K)}  # dictionary containing sets of thetas for which a is optimal
        rewards = self.R(np.arange(self.N))  # vectorized

        for theta in range(self.L):
            action_scores = np.einsum('ij,j -> i', self.q[theta, :, :], rewards)
            # for a_prime in range(self.K):
            #     expected_rewards = np.sum(self.q[theta, a_prime, :] *rewards)
            #     action_scores.append(expected_rewards)

            optimal_action = np.argmax(action_scores)
            Theta_a[optimal_action].append(theta)

        return Theta_a


    def compute_algorithm_1(self):
        # Step 1 compute Theta_a after optimal actions computation
        Theta_a = self.compute_Theta_a()

        # Step 2: Calculate the probability that each action is optimal
        p_star = np.zeros(self.K)
        for a_star in range(self.K):
            for theta in Theta_a[a_star]:
                p_star[a_star] += self.p[theta]

        # Step 3: Compute the marginal distribution of Y_1,a
        pa = np.zeros((self.K, self.N))  # AxY
        for a in range(self.K):
            for y in range(self.N):
                for theta in range(self.L):
                    pa[a, y] += self.p[theta] * self.q[theta, a, y]

        # Step 4: Compute the joint probability mass function
        pa_joint = np.zeros((self.K, self.K, self.N))
        for a in range(self.K):
            for a_star in range(self.K):
                for y in range(self.N):
                    if p_star[a_star] > 0:
                        for theta in Theta_a[a_star]:
                            pa_joint[a, a_star, y] += self.q[theta, a, y]
                        pa_joint[a, a_star, y] /= p_star[a_star]

        # Step 5: Compute R*
        R_star = 0
        rewards = self.R(np.arange(self.N))
        for a in range(self.K):
            for theta in Theta_a[a]:
                R_star += self.p[theta] * np.dot(self.q[theta, a, :], rewards)

        # Step 6: Compute g_a
        g_a = np.zeros(self.K)
        for a in range(self.K):
            for a_star in range(self.K):
                for y in range(self.N):
                    if p_star[a_star] > 0 and pa[a, y] > 0:
                        pa_joint_val = pa_joint[a, a_star, y]
                        g_a[a] += pa_joint_val * np.log(pa_joint_val / (p_star[a_star] * pa[a, y] + 1e-10))

        # Step 7: Compute delta_a
        delta = np.zeros(self.K)
        for a in range(self.K):
            expected_reward = 0
            for theta in range(self.L):
                for y in range(self.N):
                    expected_reward += self.p[theta] * self.q[theta, a, y] * self.R(y)
            delta[a] = R_star - expected_reward

        return delta, g_a

# Example usage and reward function
L, K, N = 5, 3, 4
p = np.ones(L) / L
q = np.random.rand(L, K, N)
R = lambda y: y ** 2  # Define a simple reward function
model = finiteIR(L, K, N, R, p, q)
delta, g = model.compute_algorithm_1()
print("Delta:", delta)
print("g:", g)