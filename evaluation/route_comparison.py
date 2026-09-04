from agent.route_executor import TaxiRouteExecutor
from environment.route_planner import TaxiRoutePlanner
from environment.taxi_environment import TaxiEnvironment


class RouteComparison:

    def __init__(
        self,
        q_table_path="models/q_table.npy"
    ):

        self.environment = TaxiEnvironment()

        self.executor = TaxiRouteExecutor(
            q_table_path=q_table_path
        )

        self.planner = TaxiRoutePlanner()

    # ========================================================
    # COMPARE ROUTES
    # ========================================================

    def compare(
        self,
        taxi_row,
        taxi_col,
        passenger_location,
        destination
    ):

        # ----------------------------------------------------
        # Create the same initial state for both methods
        # ----------------------------------------------------

        initial_state, info = (
            self.environment.reset_custom(
                taxi_row=taxi_row,
                taxi_col=taxi_col,
                passenger_location=passenger_location,
                destination=destination
            )
        )

        # ----------------------------------------------------
        # Q-learning route
        # ----------------------------------------------------

        rl_result = self.executor.execute(
            taxi_row=taxi_row,
            taxi_col=taxi_col,
            passenger_location=passenger_location,
            destination=destination
        )

        # ----------------------------------------------------
        # Optimal BFS route
        # ----------------------------------------------------

        optimal_route = (
            self.planner.find_shortest_route(
                initial_state
            )
        )

        # ----------------------------------------------------
        # Optimal route steps
        # ----------------------------------------------------

        if optimal_route is None:

            optimal_steps = None

        else:

            optimal_steps = len(
                optimal_route
            )

        # ----------------------------------------------------
        # RL steps
        # ----------------------------------------------------

        rl_steps = rl_result["steps"]

        # ----------------------------------------------------
        # Extra steps
        # ----------------------------------------------------

        if optimal_steps is not None:

            extra_steps = (
                rl_steps - optimal_steps
            )

        else:

            extra_steps = None

        # ----------------------------------------------------
        # Route efficiency
        # ----------------------------------------------------

        if (
            optimal_steps is not None
            and rl_steps > 0
        ):

            efficiency = (
                optimal_steps
                / rl_steps
            ) * 100

        else:

            efficiency = 0.0

        # ----------------------------------------------------
        # Optimality
        # ----------------------------------------------------

        is_optimal = (
            rl_result["success"]
            and optimal_steps is not None
            and rl_steps == optimal_steps
        )

        return {
            "initial_state": initial_state,

            "rl_result": rl_result,

            "optimal_route": optimal_route,

            "rl_steps": rl_steps,

            "optimal_steps": optimal_steps,

            "extra_steps": extra_steps,

            "route_efficiency": efficiency,

            "is_optimal": is_optimal
        }

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        self.executor.close()

        self.planner.close()

        self.environment.close()