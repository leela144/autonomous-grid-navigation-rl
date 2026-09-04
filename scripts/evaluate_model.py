import numpy as np

from environment.taxi_environment import TaxiEnvironment
from agent.q_learning import QLearningAgent


# Create environment
environment = TaxiEnvironment(render_mode="ansi")

# Create agent
agent = QLearningAgent(
    state_size=environment.state_size,
    action_size=environment.action_size
)

# Load trained Q-table
agent.q_table = np.load("models/q_table.npy")

# Disable exploration
agent.epsilon = 0.0

print("Trained Q-table loaded successfully.")
print("Q-table shape:", agent.q_table.shape)
print("Epsilon:", agent.epsilon)

# Reset environment
state, info = environment.reset()

print("\nInitial State:", state)

# Let the trained agent choose an action
action = agent.choose_action(state)

print("Selected Action:", action)

# Take the action
next_state, reward, terminated, truncated, info = (
    environment.step(action)
)

print("Next State:", next_state)
print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)

print(
    "Maximum Q-value:",
    np.max(agent.q_table)
)

print(
    "Minimum Q-value:",
    np.min(agent.q_table)
)

environment.close()