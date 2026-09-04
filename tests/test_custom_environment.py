from environment.taxi_environment import TaxiEnvironment


environment = TaxiEnvironment()


# ============================================================
# TEST USER-DEFINED TASK
# ============================================================

state, info = environment.reset_custom(
    taxi_row=0,
    taxi_col=0,
    passenger_location=0,
    destination=1
)

print("========== CUSTOM ENVIRONMENT TEST ==========")

print(
    "State:",
    state
)

print(
    "Decoded State:",
    environment.decode_state(state)
)

print(
    "Taxi Position:",
    (
        environment.decode_state(state)[0],
        environment.decode_state(state)[1]
    )
)

print(
    "Passenger:",
    environment.decode_state(state)[2]
)

print(
    "Destination:",
    environment.decode_state(state)[3]
)


# ============================================================
# TEST LANDMARKS
# ============================================================

print("\n========== LANDMARKS ==========")

for landmark_id, landmark in (
    environment.get_landmarks().items()
):

    print(
        landmark_id,
        landmark
    )


# ============================================================
# TEST WALLS
# ============================================================

print("\n========== WALL INFORMATION ==========")

walls = environment.get_walls()

print(
    "Vertical Walls:",
    len(walls["vertical"])
)

print(
    "Horizontal Walls:",
    len(walls["horizontal"])
)

print(
    "\nVertical Wall Details:"
)

for wall in walls["vertical"]:

    print(wall)

print(
    "\nHorizontal Wall Details:"
)

for wall in walls["horizontal"]:

    print(wall)


# ============================================================
# TEST MAP
# ============================================================

print(
    "\n========== ORIGINAL TAXI-v4 MAP =========="
)

print(
    environment.get_map_description()
)


environment.close()

print(
    "\nCustom environment test completed successfully."
)