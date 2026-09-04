from environment.taxi_environment import TaxiEnvironment


environment = TaxiEnvironment(render_mode="ansi")

state, info = environment.reset()

print("Initial State:", state)
print("Total States:", environment.state_size)
print("Total Actions:", environment.action_size)

print("\nEnvironment:")
print(environment.render())

environment.close()