import numpy as np
import pandas as pd

from environment.taxi_environment import TaxiEnvironment
from agent.q_learning import QLearningAgent


CONFIGURATIONS = [
    {
        "name": "Baseline",
        "learning_rate": 0.7,
        "discount_factor": 0.95,
        "epsilon_decay": 0.995
    },
    {
        "name": "Fast Learning",
        "learning_rate": 0.9,
        "discount_factor": 0.95,
        "epsilon_decay": 0.995
    },
    {
        "name": "Slow Exploration Decay",
        "learning_rate": 0.7,
        "discount_factor": 0.95,
        "epsilon_decay": 0.999
    }
]


def train_and_evaluate(config):

    environment = TaxiEnvironment()

    agent = QLearningAgent(
        state_size=environment.state_size,
        action_size=environment.action_size,
        learning_rate=config["learning_rate"],
        discount_factor=config["discount_factor"],
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=config["epsilon_decay"]
    )

    training_episodes = 5000
    max_steps = 200

    for episode in range(training_episodes):

        state, info = environment.reset()

        for step in range(max_steps):

            action = agent.choose_action(state)

            next_state, reward, terminated, truncated, info = (
                environment.step(action)
            )

            agent.update(
                state,
                action,
                reward,
                next_state,
                terminated
            )

            state = next_state

            if terminated or truncated:
                break

        agent.decay_epsilon()

    environment.close()

    # Evaluation
    evaluation_environment = TaxiEnvironment()

    agent.epsilon = 0.0

    evaluation_episodes = 500

    rewards = []
    steps_list = []
    successful = 0

    for episode in range(evaluation_episodes):

        state, info = evaluation_environment.reset()

        total_reward = 0
        steps = 0
        terminated = False

        for step in range(max_steps):

            action = agent.choose_action(state)

            next_state, reward, terminated, truncated, info = (
                evaluation_environment.step(action)
            )

            state = next_state

            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        rewards.append(total_reward)
        steps_list.append(steps)

        if terminated:
            successful += 1

    evaluation_environment.close()

    return {
        "Configuration": config["name"],
        "Learning Rate": config["learning_rate"],
        "Discount Factor": config["discount_factor"],
        "Epsilon Decay": config["epsilon_decay"],
        "Average Reward": np.mean(rewards),
        "Average Steps": np.mean(steps_list),
        "Success Rate": (successful / evaluation_episodes) * 100
    }


results = []

for config in CONFIGURATIONS:

    print(
        f"\nRunning configuration: "
        f"{config['name']}"
    )

    result = train_and_evaluate(config)

    results.append(result)

    print(
        f"Reward: {result['Average Reward']:.2f} | "
        f"Steps: {result['Average Steps']:.2f} | "
        f"Success: {result['Success Rate']:.2f}%"
    )


results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/hyperparameter_results.csv",
    index=False
)

print("\n========== HYPERPARAMETER RESULTS ==========")
print(results_df.to_string(index=False))
print("\nResults saved to:")
print("results/hyperparameter_results.csv")