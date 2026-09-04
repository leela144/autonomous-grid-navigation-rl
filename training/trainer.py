import numpy as np

from environment.taxi_environment import TaxiEnvironment
from agent.q_learning import QLearningAgent


class Trainer:

    def __init__(
        self,
        episodes=10000,
        max_steps=200
    ):
        self.episodes = episodes
        self.max_steps = max_steps

        # Create environment
        self.environment = TaxiEnvironment()

        # Create Q-learning agent
        self.agent = QLearningAgent(
            state_size=self.environment.state_size,
            action_size=self.environment.action_size
        )

        # Store training statistics
        self.rewards = []
        self.steps_per_episode = []
        self.epsilon_history = []

    def train(self):

        for episode in range(self.episodes):

            state, info = self.environment.reset()

            total_reward = 0
            steps = 0

            for step in range(self.max_steps):

                # Choose action
                action = self.agent.choose_action(state)

                # Perform action
                next_state, reward, terminated, truncated, info = (
                    self.environment.step(action)
                )

                # Update Q-table
                self.agent.update(
                    state,
                    action,
                    reward,
                    next_state,
                    terminated
                )

                state = next_state

                total_reward += reward
                steps += 1

                # Stop if episode is finished
                if terminated or truncated:
                    break

            # Reduce exploration
            self.agent.decay_epsilon()

            # Store statistics
            self.rewards.append(total_reward)
            self.steps_per_episode.append(steps)
            self.epsilon_history.append(self.agent.epsilon)

            # Progress information
            if (episode + 1) % 1000 == 0:
                recent_rewards = self.rewards[-1000:]
                average_reward = np.mean(recent_rewards)

                print(
                    f"Episode: {episode + 1}/{self.episodes} | "
                    f"Average Reward: {average_reward:.2f} | "
                    f"Epsilon: {self.agent.epsilon:.4f}"
                )

        self.environment.close()

        return {
            "rewards": self.rewards,
            "steps": self.steps_per_episode,
            "epsilon": self.epsilon_history,
            "q_table": self.agent.q_table
        }