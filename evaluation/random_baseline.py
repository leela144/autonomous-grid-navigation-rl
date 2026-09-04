import numpy as np

from environment.taxi_environment import TaxiEnvironment


class RandomAgentEvaluator:

    def __init__(
        self,
        episodes=1000,
        max_steps=200
    ):
        self.episodes = episodes
        self.max_steps = max_steps

        self.environment = TaxiEnvironment()

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

                action = self.environment.env.action_space.sample()

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
            )
        }