class PerformanceMetrics:

    @staticmethod
    def calculate_improvement(
        baseline_value,
        trained_value,
        metric_type="higher"
    ):
        if metric_type == "higher":
            return (
                (trained_value - baseline_value)
                / abs(baseline_value)
            ) * 100

        return (
            (baseline_value - trained_value)
            / abs(baseline_value)
        ) * 100

    @staticmethod
    def generate_report(
        baseline,
        trained
    ):

        reward_improvement = (
            trained["average_reward"]
            - baseline["average_reward"]
        )

        steps_reduction = (
            (
                baseline["average_steps"]
                - trained["average_steps"]
            )
            / baseline["average_steps"]
        ) * 100

        success_improvement = (
            trained["success_rate"]
            - baseline["success_rate"]
        )

        return {
            "reward_improvement": reward_improvement,
            "steps_reduction_percent": steps_reduction,
            "success_rate_improvement": success_improvement
        }