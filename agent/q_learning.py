import numpy as np


class QLearningAgent:

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.7,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    ):
        self.state_size = state_size
        self.action_size = action_size

        # Q-learning hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Exploration parameters
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table
        self.q_table = np.zeros(
            (state_size, action_size)
        )

    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy strategy.
        """

        if np.random.random() < self.epsilon:
            # Explore
            return np.random.randint(self.action_size)

        # Exploit
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, terminated):
        """
        Update Q-value using the Q-learning formula.
        """

        current_q = self.q_table[state, action]

        if terminated:
            target_q = reward
        else:
            best_next_q = np.max(self.q_table[next_state])
            target_q = (
                reward
                + self.discount_factor * best_next_q
            )

        new_q = current_q + self.learning_rate * (
            target_q - current_q
        )

        self.q_table[state, action] = new_q

    def decay_epsilon(self):
        """
        Reduce exploration after each episode.
        """

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )