import numpy as np

from visualization.plots import TrainingVisualizer


rewards = np.load("results/rewards.npy")
steps = np.load("results/steps.npy")
epsilon = np.load("results/epsilon.npy")


print("\n========== TRAINING ANALYSIS ==========")

print(f"Total Episodes: {len(rewards)}")

print(
    f"Initial Average Reward "
    f"(First 100): {np.mean(rewards[:100]):.2f}"
)

print(
    f"Final Average Reward "
    f"(Last 100): {np.mean(rewards[-100:]):.2f}"
)

print(
    f"Initial Average Steps "
    f"(First 100): {np.mean(steps[:100]):.2f}"
)

print(
    f"Final Average Steps "
    f"(Last 100): {np.mean(steps[-100:]):.2f}"
)

print(
    f"Final Epsilon: {epsilon[-1]:.4f}"
)

print("=======================================")


TrainingVisualizer.plot_rewards(rewards)
TrainingVisualizer.plot_steps(steps)
TrainingVisualizer.plot_epsilon(epsilon)
