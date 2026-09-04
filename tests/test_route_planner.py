from environment.taxi_environment import TaxiEnvironment
from environment.route_planner import TaxiRoutePlanner


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

environment = TaxiEnvironment()


# ============================================================
# CREATE CUSTOM TASK
# ============================================================

state, info = environment.reset_custom(
    taxi_row=0,
    taxi_col=0,
    passenger_location=0,
    destination=1
)


print("========== TAXI ROUTE PLANNER TEST ==========")

print(
    "Initial State:",
    state
)

print(
    "Decoded State:",
    environment.decode_state(state)
)


# ============================================================
# CREATE ROUTE PLANNER
# ============================================================

planner = TaxiRoutePlanner()


# ============================================================
# FIND SHORTEST ROUTE
# ============================================================

route = planner.find_shortest_route(
    state
)


print("\n========== SHORTEST ROUTE ==========")

print(
    "Route:",
    route
)

print(
    "Number of Actions:",
    len(route)
)


# ============================================================
# ACTION NAMES
# ============================================================

route_names = planner.get_route_names(
    route
)


print(
    "\nRoute Actions:"
)

for index, action in enumerate(route_names, start=1):

    print(
        f"{index}. {action}"
    )


# ============================================================
# CLEANUP
# ============================================================

planner.close()
environment.close()


print(
    "\nRoute planner test completed successfully."
)