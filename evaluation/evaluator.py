import numpy as np

from environment.taxi_environment import TaxiEnvironment
from agent.q_learning import QLearningAgent


class Evaluator:

    def __init__(
        self,
        model_path="models/q_table.npy",
        episodes=1000,
        max_steps=200
    ):
        self.episodes = episodes
        self.max_steps = max_steps

        # Create environment
        self.environment = TaxiEnvironment()

        # Create agent
        self.agent = QLearningAgent(
            state_size=self.environment.state_size,
            action_size=self.environment.action_size
        )

        # Load trained Q-table
        self.agent.q_table = np.load(model_path)

        # Disable exploration
        self.agent.epsilon = 0.0

    def evaluate(self):

        rewards = []
        steps_list = []

        successful_episodes = 0

        for episode in range(self.episodes):

            state, info = self.environment.reset()

            total_reward = 0
            steps = 0
            terminated = False

            for step in range(self.max_steps):

                # Always choose the learned best action
                action = self.agent.choose_action(state)

                next_state, reward, terminated, truncated, info = (
                    self.environment.step(action)
                )

                state = next_state

                total_reward += reward
                steps += 1

                if terminated or truncated:
                    break

            rewards.append(total_reward)
            steps_list.append(steps)

            if terminated:
                successful_episodes += 1

        self.environment.close()

        return {
            "average_reward": np.mean(rewards),
            "average_steps": np.mean(steps_list),
            "success_rate": (
                successful_episodes / self.episodes
            ) * 100,
            "successful_episodes": successful_episodes,
            "failed_episodes": (
                self.episodes - successful_episodes
            ),
            "rewards": rewards,
            "steps": steps_list
        }