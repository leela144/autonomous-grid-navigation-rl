from evaluation.evaluator import Evaluator


evaluator = Evaluator(
    model_path="models/q_table.npy",
    episodes=1000,
    max_steps=200
)

results = evaluator.evaluate()


print("\n========== EVALUATION RESULTS ==========")

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

print("=========================================")