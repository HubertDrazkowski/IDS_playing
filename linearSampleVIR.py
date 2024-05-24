import numpy as np
import matplotlib.pyplot as plt


def linearSampleVIR(K, d, M, theta_samples, Phi):
    # Estimating μ based on the sample mean of theta
    mu_hat = np.mean(theta_samples, axis=0)

    # Finding which action is best for each theta sample based on Phi^T * theta
    best_action_indices = np.argmax(Phi.T @ theta_samples.T, axis=0)

    # Calculate p*(a) - probability each action is best
    p_star = np.array([np.sum(best_action_indices == a) for a in range(K)]) / M

    # Calculate μa for each action
    mu_a_hat = np.array(
        [np.mean(theta_samples[best_action_indices == a], axis=0) if np.any(best_action_indices == a) else np.zeros(d)
         for a in range(K)])

    # Calculate L
    L_hat = sum(p_star[a] * np.outer(mu_a_hat[a] - mu_hat, mu_a_hat[a] - mu_hat) for a in range(K))

    # Calculate rho*
    rho_star = sum(p_star[a] * (Phi[:, a].T @ mu_a_hat[a]) for a in range(K))

    # Calculate va and Delta_a
    v_a = np.array([Phi[:, a].T @ L_hat @ Phi[:, a] for a in range(K)])
    Delta_a = rho_star - np.array([Phi[:, a].T @ mu_hat for a in range(K)])

    return Delta_a, v_a




# Setting up the simulation parameters
K = 30  # Number of actions
d = 5  # Dimensionality of feature vectors
N = 2000  # Number of trials
theta_true = np.random.multivariate_normal(np.zeros(d), 10 * np.eye(d))  # True theta from Gaussian prior

# Generate the action set with features uniformly distributed in [-1/sqrt(d), 1/sqrt(d)]
actions = np.random.uniform(-1 / np.sqrt(d), 1 / np.sqrt(d), (K, d))


# Function to simulate the reward for an action
def simulate_reward(action, theta, noise_var=1):
    noise = np.random.normal(0, np.sqrt(noise_var))
    return action @ theta + noise


# Function to run the simulation over N trials
def run_simulation(actions, theta, N):
    regrets = np.zeros(N)
    rewards = np.zeros(N)
    for i in range(N):
        # Randomly select an action
        action_index = np.random.randint(0, K)
        selected_action = actions[action_index]

        # Compute the reward
        reward = simulate_reward(selected_action, theta)
        rewards[i] = reward

        # Compute optimal reward
        optimal_rewards = [simulate_reward(a, theta) for a in actions]
        optimal_reward = max(optimal_rewards)

        # Calculate regret
        regrets[i] = optimal_reward - reward

    return regrets, rewards


# Run the simulation
regrets, rewards = run_simulation(actions, theta_true, N)

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(regrets), label='Cumulative Regret')
plt.title('Cumulative Regret over Time')
plt.xlabel('Trial')
plt.ylabel('Cumulative Regret')
plt.legend()
plt.show()

# Plot rewards
plt.figure(figsize=(10, 5))
plt.plot(rewards, label='Rewards per Trial')
plt.title('Rewards over Trials')
plt.xlabel('Trial')
plt.ylabel('Reward')
plt.legend()
plt.show()
