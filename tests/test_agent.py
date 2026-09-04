from environment.taxi_environment import TaxiEnvironment
from agent.q_learning import QLearningAgent


environment = TaxiEnvironment()

agent = QLearningAgent(
    state_size=environment.state_size,
    action_size=environment.action_size
)

print("State Size:", agent.state_size)
print("Action Size:", agent.action_size)
print("Q-Table Shape:", agent.q_table.shape)
print("Initial Epsilon:", agent.epsilon)

state, info = environment.reset()

action = agent.choose_action(state)

print("Initial State:", state)
print("Selected Action:", action)

environment.close()