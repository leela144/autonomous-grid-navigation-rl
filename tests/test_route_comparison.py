from evaluation.route_comparison import RouteComparison


# ============================================================
# CREATE COMPARISON SYSTEM
# ============================================================

comparison = RouteComparison()


# ============================================================
# CUSTOM TASK
# ============================================================

result = comparison.compare(
    taxi_row=0,
    taxi_col=0,
    passenger_location=0,
    destination=1
)


# ============================================================
# RESULTS
# ============================================================

print(
    "========== ROUTE COMPARISON =========="
)

print(
    "RL Agent Steps:",
    result["rl_steps"]
)

print(
    "Optimal Steps:",
    result["optimal_steps"]
)

print(
    "Extra Steps:",
    result["extra_steps"]
)

print(
    "Route Efficiency:",
    f"{result['route_efficiency']:.2f}%"
)

print(
    "RL Success:",
    result["rl_result"]["success"]
)

print(
    "RL Route Optimal:",
    result["is_optimal"]
)


# ============================================================
# RL ACTIONS
# ============================================================

print(
    "\n========== RL ACTIONS =========="
)

rl_actions = result[
    "rl_result"
]["actions"]

for index, action in enumerate(
    rl_actions,
    start=1
):

    print(
        f"{index}. {action}"
    )


# ============================================================
# OPTIMAL ACTIONS
# ============================================================

print(
    "\n========== OPTIMAL ACTIONS =========="
)

optimal_route = result[
    "optimal_route"
]

if optimal_route is None:

    print(
        "No valid route found."
    )

else:

    for index, action in enumerate(
        optimal_route,
        start=1
    ):

        print(
            f"{index}. {action}"
        )


# ============================================================
# CLEANUP
# ============================================================

comparison.close()


print(
    "\nRoute comparison test completed successfully."
)