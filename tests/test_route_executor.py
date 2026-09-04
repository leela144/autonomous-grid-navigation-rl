from agent.route_executor import TaxiRouteExecutor


# ============================================================
# CREATE EXECUTOR
# ============================================================

executor = TaxiRouteExecutor(
    q_table_path="models/q_table.npy"
)


# ============================================================
# CUSTOM TASK
# ============================================================

result = executor.execute(
    taxi_row=0,
    taxi_col=0,
    passenger_location=0,
    destination=1
)


# ============================================================
# RESULTS
# ============================================================

print(
    "========== Q-LEARNING ROUTE EXECUTION =========="
)

print(
    "Initial State:",
    result["initial_state"]
)

print(
    "Final State:",
    result["final_state"]
)

print(
    "Steps:",
    result["steps"]
)

print(
    "Total Reward:",
    result["total_reward"]
)

print(
    "Success:",
    result["success"]
)


# ============================================================
# ACTIONS
# ============================================================

print(
    "\n========== ACTION SEQUENCE =========="
)

action_names = executor.get_action_names(
    result["actions"]
)

for index, action in enumerate(
    action_names,
    start=1
):

    print(
        f"{index}. {action}"
    )


# ============================================================
# POSITIONS
# ============================================================

print(
    "\n========== TAXI POSITIONS =========="
)

for index, position in enumerate(
    result["positions"]
):

    print(
        f"{index}: {position}"
    )


# ============================================================
# CLEANUP
# ============================================================

executor.close()


print(
    "\nQ-learning route execution test completed successfully."
)