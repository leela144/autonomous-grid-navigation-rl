import numpy as np

from environment.taxi_environment import TaxiEnvironment


class TaxiRouteExecutor:

    def __init__(
        self,
        q_table_path="models/q_table.npy",
        max_steps=200
    ):

        self.environment = TaxiEnvironment()

        self.q_table = np.load(q_table_path)

        self.max_steps = max_steps

    # ========================================================
    # EXECUTE LEARNED ROUTE
    # ========================================================

    def execute(
        self,
        taxi_row,
        taxi_col,
        passenger_location,
        destination
    ):

        # Create the requested Taxi-v4 state
        state, info = self.environment.reset_custom(
            taxi_row=taxi_row,
            taxi_col=taxi_col,
            passenger_location=passenger_location,
            destination=destination
        )

        states = [state]
        actions = []
        rewards = []
        positions = []

        total_reward = 0

        # Initial taxi position
        decoded = self.environment.decode_state(state)

        current_position = (
            decoded[0],
            decoded[1]
        )

        positions.append(current_position)

        terminated = False
        truncated = False

        # ====================================================
        # FOLLOW Q-TABLE
        # ====================================================

        for _ in range(self.max_steps):

            # Select the action with the highest Q-value
            action = int(
                np.argmax(self.q_table[state])
            )

            next_state, reward, terminated, truncated, info = (
                self.environment.step(action)
            )

            # Store action and reward
            actions.append(action)
            rewards.append(reward)
            states.append(next_state)

            total_reward += reward

            # Get new taxi position
            decoded = self.environment.decode_state(
                next_state
            )

            next_position = (
                decoded[0],
                decoded[1]
            )

            # Store only actual taxi movements.
            #
            # If Taxi-v4 blocks a movement because of a wall,
            # the taxi remains in the same cell.
            #
            # Therefore no fake route segment is created.
            if next_position != current_position:

                positions.append(next_position)

                current_position = next_position

            state = next_state

            # Stop when Taxi-v4 reaches the terminal state
            if terminated or truncated:
                break

        return {
            "initial_state": states[0],
            "final_state": state,
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "positions": positions,
            "total_reward": total_reward,
            "steps": len(actions),
            "success": terminated
        }

    # ========================================================
    # CONVERT ACTION NUMBERS TO NAMES
    # ========================================================

    def get_action_names(self, actions):

        return [
            self.environment.get_action_name(action)
            for action in actions
        ]

    # ========================================================
    # CLOSE ENVIRONMENT
    # ========================================================

    def close(self):

        self.environment.close()