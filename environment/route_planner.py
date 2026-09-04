from collections import deque

from environment.taxi_environment import TaxiEnvironment


class TaxiRoutePlanner:

    def __init__(self):

        self.environment = TaxiEnvironment()

    # ========================================================
    # FIND SHORTEST ROUTE
    # ========================================================

    def find_shortest_route(self, initial_state):

        """
        Find the shortest valid action sequence from the
        given Taxi-v4 state to successful passenger dropoff.

        BFS is used only as an optimal-route benchmark.
        It does NOT control the Q-learning agent.
        """

        queue = deque()

        queue.append(
            (
                initial_state,
                []
            )
        )

        visited = set()

        visited.add(initial_state)

        # ----------------------------------------------------
        # Actions
        # ----------------------------------------------------

        actions = [
            0,  # South
            1,  # North
            2,  # East
            3,  # West
            4,  # Pickup
            5   # Dropoff
        ]

        # ----------------------------------------------------
        # BFS
        # ----------------------------------------------------

        while queue:

            state, route = queue.popleft()

            # ------------------------------------------------
            # Decode current state
            # ------------------------------------------------

            taxi_row, taxi_col, passenger, destination = (
                self.environment.decode_state(state)
            )

            # ------------------------------------------------
            # Goal condition
            #
            # Passenger has reached destination.
            # In Taxi-v4 the passenger location becomes
            # the destination after successful dropoff.
            # ------------------------------------------------

            if (
                passenger == destination
                and len(route) > 0
            ):

                return route

            # ------------------------------------------------
            # Explore actions
            # ------------------------------------------------

            for action in actions:

                # Get Taxi-v4 transition information
                transitions = (
                    self.environment.env.unwrapped.P[state][action]
                )

                next_state = transitions[0][1]
                reward = transitions[0][2]
                terminated = transitions[0][3]

                # ------------------------------------------------
                # Ignore transitions that don't change state
                # ------------------------------------------------

                if next_state == state and not terminated:
                    continue

                # ------------------------------------------------
                # Avoid revisiting states
                # ------------------------------------------------

                if next_state in visited:
                    continue

                visited.add(next_state)

                new_route = route + [action]

                # ------------------------------------------------
                # Successful terminal state
                # ------------------------------------------------

                if terminated:

                    return new_route

                queue.append(
                    (
                        next_state,
                        new_route
                    )
                )

        return None

    # ========================================================
    # ACTION NAMES
    # ========================================================

    def get_route_names(self, route):

        if route is None:
            return []

        return [
            self.environment.get_action_name(action)
            for action in route
        ]

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        self.environment.close()