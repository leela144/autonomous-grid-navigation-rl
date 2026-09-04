from evaluation.random_baseline import RandomAgentEvaluator


evaluator = RandomAgentEvaluator(
    episodes=1000,
    max_steps=200
)

results = evaluator.evaluate()


print("\n====== RANDOM AGENT BASELINE ======")

print(
    f"Average Reward: "
    f"{results['average_reward']:.2f}"
)

print(
    f"Average Steps: "
    f"{results['average_steps']:.2f}"
)

print(
    f"Success Rate: "
    f"{results['success_rate']:.2f}%"
)

print(
    f"Successful Episodes: "
    f"{results['successful_episodes']}"
)

print(
    f"Failed Episodes: "
    f"{results['failed_episodes']}"
)

print("===================================")