from evaluation.evaluator import Evaluator
from evaluation.random_baseline import RandomAgentEvaluator
from evaluation.metrics import PerformanceMetrics


episodes = 1000
max_steps = 200


trained_agent = Evaluator(
    model_path="models/q_table.npy",
    episodes=episodes,
    max_steps=max_steps
)

random_agent = RandomAgentEvaluator(
    episodes=episodes,
    max_steps=max_steps
)


trained_results = trained_agent.evaluate()
random_results = random_agent.evaluate()


metrics = PerformanceMetrics.generate_report(
    baseline=random_results,
    trained=trained_results
)


print("\n========== AGENT COMPARISON ==========")

print(
    f"Reward Improvement: "
    f"{metrics['reward_improvement']:.2f}"
)

print(
    f"Steps Reduction: "
    f"{metrics['steps_reduction_percent']:.2f}%"
)

print(
    f"Success Rate Improvement: "
    f"{metrics['success_rate_improvement']:.2f} percentage points"
)

print("======================================")