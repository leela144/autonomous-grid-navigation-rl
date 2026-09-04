import gymnasium as gym


class TaxiEnvironment:

    # ========================================================
    # TAXI-v4 LANDMARKS
    # ========================================================

    LANDMARKS = {
        0: {
            "name": "Red",
            "position": (0, 0),
            "symbol": "R"
        },
        1: {
            "name": "Green",
            "position": (0, 4),
            "symbol": "G"
        },
        2: {
            "name": "Yellow",
            "position": (4, 0),
            "symbol": "Y"
        },
        3: {
            "name": "Blue",
            "position": (4, 3),
            "symbol": "B"
        }
    }

    # Taxi-v4 action mapping
    ACTIONS = {
        0: "South",
        1: "North",
        2: "East",
        3: "West",
        4: "Pickup",
        5: "Dropoff"
    }

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self, render_mode=None):

        self.env = gym.make(
            "Taxi-v4",
            render_mode=render_mode
        )

        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        state, info = self.env.reset()

        return state, info

    # ========================================================
    # CUSTOM RESET
    # ========================================================

    def reset_custom(
        self,
        taxi_row,
        taxi_col,
        passenger_location,
        destination
    ):

        # Validate taxi position
        if not (0 <= taxi_row <= 4):
            raise ValueError("taxi_row must be between 0 and 4")

        if not (0 <= taxi_col <= 4):
            raise ValueError("taxi_col must be between 0 and 4")

        # Validate passenger landmark
        if not (0 <= passenger_location <= 3):
            raise ValueError(
                "passenger_location must be between 0 and 3"
            )

        # Validate destination landmark
        if not (0 <= destination <= 3):
            raise ValueError(
                "destination must be between 0 and 3"
            )

        # Passenger and destination cannot be the same
        if passenger_location == destination:
            raise ValueError(
                "Passenger location and destination must be different"
            )

        # Reset environment first
        self.env.reset()

        # Create Taxi-v4 state
        state = self.env.unwrapped.encode(
            taxi_row,
            taxi_col,
            passenger_location,
            destination
        )

        # Directly set the environment state
        self.env.unwrapped.s = state

        return state, {}

    # ========================================================
    # STEP
    # ========================================================

    def step(self, action):

        if not (0 <= action < self.action_size):
            raise ValueError(
                f"Action must be between 0 and {self.action_size - 1}"
            )

        return self.env.step(action)

    # ========================================================
    # RENDER
    # ========================================================

    def render(self):

        return self.env.render()

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        self.env.close()

    # ========================================================
    # STATE DECODING
    # ========================================================

    def decode_state(self, state):

        return self.env.unwrapped.decode(state)

    # ========================================================
    # STATE ENCODING
    # ========================================================

    def encode_state(
        self,
        taxi_row,
        taxi_col,
        passenger_location,
        destination
    ):

        return self.env.unwrapped.encode(
            taxi_row,
            taxi_col,
            passenger_location,
            destination
        )

    # ========================================================
    # LANDMARK INFORMATION
    # ========================================================

    def get_landmarks(self):

        return self.LANDMARKS

    # ========================================================
    # ACTION INFORMATION
    # ========================================================

    def get_action_name(self, action):

        return self.ACTIONS.get(
            action,
            "Unknown"
        )

    # ========================================================
    # MAP DESCRIPTION
    # ========================================================

    def get_map_description(self):

        desc = self.env.unwrapped.desc

        rows = []

        for row in desc:

            decoded_row = []

            for character in row:

                if isinstance(character, bytes):
                    decoded_row.append(
                        character.decode("utf-8")
                    )
                else:
                    decoded_row.append(
                        chr(character)
                    )

            rows.append(
                "".join(decoded_row)
            )

        return "\n".join(rows)

    # ========================================================
    # WALL INFORMATION
    # ========================================================

    def get_walls(self):

        """
        Return the actual internal walls of Taxi-v4.

        Taxi-v4 represents its 5x5 city using an ASCII map.
        Internal movement barriers are vertical walls between
        adjacent columns.

        The official Taxi-v4 movement implementation checks:

            self.desc[1 + row, 2 * col + 2] == b":"

        for East movement.

        Therefore, if that position is not ':', an East movement
        is blocked by a wall.

        Taxi-v4 does not have internal horizontal walls.
        North/South movement only uses the grid boundaries.
        """

        vertical_walls = []

        desc = self.env.unwrapped.desc

        # There are 5 rows and 5 columns.
        #
        # Check the boundary between:
        #
        # column 0 <-> 1
        # column 1 <-> 2
        # column 2 <-> 3
        # column 3 <-> 4
        #
        # for every row.

        for row in range(5):

            for col in range(4):

                # This is exactly the position Taxi-v4
                # checks before allowing East movement.
                wall_position = desc[
                    1 + row,
                    2 * col + 2
                ]

                # ':' means movement is allowed.
                # Anything else means a wall exists.
                if wall_position != b":":

                    vertical_walls.append(
                        {
                            "row": row,
                            "left_col": col,
                            "right_col": col + 1
                        }
                    )

        # Taxi-v4 has no internal horizontal walls.
        horizontal_walls = []

        return {
            "vertical": vertical_walls,
            "horizontal": horizontal_walls
        }

    # ========================================================
    # COMPLETE MAP INFORMATION
    # ========================================================

    def get_map_info(self):

        return {
            "rows": 5,
            "columns": 5,
            "landmarks": self.get_landmarks(),
            "walls": self.get_walls(),
            "description": self.get_map_description()
        }