from environment.taxi_environment import TaxiEnvironment
from visualization.plots import create_taxi_map


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

environment = TaxiEnvironment()


# ============================================================
# CUSTOM TASK
# ============================================================

state, info = environment.reset_custom(
    taxi_row=0,
    taxi_col=0,
    passenger_location=0,
    destination=1
)


# ============================================================
# DECODE STATE
# ============================================================

taxi_row, taxi_col, passenger, destination = (
    environment.decode_state(state)
)


# ============================================================
# GET POSITIONS
# ============================================================

taxi_position = (
    taxi_row,
    taxi_col
)


passenger_position = (
    environment.get_landmarks()[
        passenger
    ]["position"]
)


destination_position = (
    environment.get_landmarks()[
        destination
    ]["position"]
)


# ============================================================
# GET WALLS
# ============================================================

walls = environment.get_walls()


# ============================================================
# CREATE MAP
# ============================================================

figure = create_taxi_map(
    taxi_position=taxi_position,
    passenger_position=passenger_position,
    destination_position=destination_position,
    walls=walls,
    title="Taxi-v4 Custom Navigation"
)


# ============================================================
# DISPLAY
# ============================================================

figure.show()


print(
    "Visualization test completed successfully."
)


# ============================================================
# CLEANUP
# ============================================================

environment.close()