import streamlit as st
import matplotlib.pyplot as plt

from environment.taxi_environment import TaxiEnvironment
from evaluation.route_comparison import RouteComparison


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Taxi Route Optimization",
    page_icon="🚕",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #666;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "result" not in st.session_state:
    st.session_state.result = None


# ============================================================
# LOAD TAXI ENVIRONMENT
# ============================================================

@st.cache_resource
def load_environment():
    return TaxiEnvironment()


environment = load_environment()
landmarks = environment.get_landmarks()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚕 Taxi Route Optimization</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Reinforcement Learning based route planning using
        Q-Learning on the Gymnasium Taxi-v4 environment.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🎯 Define Taxi Task")

st.sidebar.markdown(
    "Configure the taxi starting position, passenger pickup "
    "location, and destination."
)


# ============================================================
# TAXI START
# ============================================================

st.sidebar.subheader("🚕 Taxi Start")

taxi_row = st.sidebar.selectbox(
    "Row",
    options=list(range(5)),
    index=0
)

taxi_col = st.sidebar.selectbox(
    "Column",
    options=list(range(5)),
    index=0
)


# ============================================================
# LANDMARKS
# ============================================================

landmark_names = {
    landmark_id:
    f"{landmark['symbol']} {landmark['name']} "
    f"({landmark['position'][0]}, {landmark['position'][1]})"
    for landmark_id, landmark in landmarks.items()
}


# ============================================================
# PASSENGER
# ============================================================

st.sidebar.subheader("🧍 Passenger")

passenger_id = st.sidebar.selectbox(
    "Pickup Location",
    options=list(landmarks.keys()),
    format_func=lambda x: landmark_names[x]
)


# ============================================================
# DESTINATION
# ============================================================

st.sidebar.subheader("🎯 Destination")

destination_options = [
    landmark_id
    for landmark_id in landmarks.keys()
    if landmark_id != passenger_id
]

destination_id = st.sidebar.selectbox(
    "Dropoff Location",
    options=destination_options,
    format_func=lambda x: landmark_names[x]
)


# ============================================================
# TASK SUMMARY
# ============================================================

st.sidebar.markdown("---")

st.sidebar.markdown("**Selected Task**")

st.sidebar.write(
    f"Taxi: ({taxi_row}, {taxi_col})"
)

st.sidebar.write(
    f"Passenger: {landmarks[passenger_id]['name']}"
)

st.sidebar.write(
    f"Destination: {landmarks[destination_id]['name']}"
)


# ============================================================
# START NAVIGATION
# ============================================================

start_navigation = st.sidebar.button(
    "🚀 Start Navigation",
    use_container_width=True
)


# ============================================================
# RUN Q-LEARNING AGENT
# ============================================================

if start_navigation:

    with st.spinner("Running Q-Learning agent..."):

        comparison = RouteComparison()

        try:

            result = comparison.compare(
                taxi_row=taxi_row,
                taxi_col=taxi_col,
                passenger_location=passenger_id,
                destination=destination_id
            )

            st.session_state.result = result

        finally:

            comparison.close()


# ============================================================
# MAP STATE
# ============================================================

if st.session_state.result is None:

    current_taxi_position = (
        taxi_row,
        taxi_col
    )

    route = None

else:

    result = st.session_state.result

    positions = result["rl_result"]["positions"]

    current_taxi_position = positions[-1]

    route = positions


# ============================================================
# PASSENGER / DESTINATION POSITIONS
# ============================================================

passenger_position = landmarks[
    passenger_id
]["position"]

destination_position = landmarks[
    destination_id
]["position"]


# ============================================================
# CREATE TAXI MAP
# ============================================================

def create_dashboard_map(
    taxi_position,
    passenger_position,
    destination_position,
    walls,
    route=None
):

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    # --------------------------------------------------------
    # MAP LIMITS
    # --------------------------------------------------------

    ax.set_xlim(0, 5)
    ax.set_ylim(5, 0)
    ax.set_aspect("equal")

    # --------------------------------------------------------
    # GRID
    # --------------------------------------------------------

    for x in range(6):

        ax.plot(
            [x, x],
            [0, 5],
            linewidth=0.8,
            alpha=0.25,
            zorder=1
        )

    for y in range(6):

        ax.plot(
            [0, 5],
            [y, y],
            linewidth=0.8,
            alpha=0.25,
            zorder=1
        )

    # --------------------------------------------------------
    # ROUTE
    # --------------------------------------------------------

    if route is not None and len(route) > 1:

        route_xy = [
            (
                position[1] + 0.5,
                position[0] + 0.5
            )
            for position in route
        ]

        x_values = [
            point[0]
            for point in route_xy
        ]

        y_values = [
            point[1]
            for point in route_xy
        ]

        ax.plot(
            x_values,
            y_values,
            linewidth=3,
            marker="o",
            markersize=4,
            alpha=0.65,
            zorder=2
        )

    # --------------------------------------------------------
    # LANDMARKS
    # --------------------------------------------------------

    landmark_markers = {
        0: "R",
        1: "G",
        2: "Y",
        3: "B"
    }

    for landmark_id, landmark in landmarks.items():

        row, col = landmark["position"]

        ax.text(
            col + 0.15,
            row + 0.18,
            landmark_markers[landmark_id],
            fontsize=11,
            fontweight="bold",
            zorder=4
        )

        ax.text(
            col + 0.15,
            row + 0.78,
            landmark["name"],
            fontsize=7,
            zorder=4
        )

    # --------------------------------------------------------
    # PASSENGER
    # --------------------------------------------------------

    row, col = passenger_position

    ax.scatter(
        col + 0.5,
        row + 0.5,
        s=350,
        marker="o",
        facecolors="none",
        linewidths=2,
        zorder=4
    )

    ax.text(
        col + 0.5,
        row + 0.5,
        "P",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        zorder=5
    )

    # --------------------------------------------------------
    # DESTINATION
    # --------------------------------------------------------

    row, col = destination_position

    ax.scatter(
        col + 0.5,
        row + 0.5,
        s=450,
        marker="*",
        linewidths=1.5,
        zorder=4
    )

    ax.text(
        col + 0.5,
        row + 0.82,
        "DEST",
        ha="center",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        zorder=5
    )

    # --------------------------------------------------------
    # TAXI
    # --------------------------------------------------------

    row, col = taxi_position

    ax.scatter(
        col + 0.5,
        row + 0.5,
        s=500,
        marker="s",
        linewidths=2,
        zorder=5
    )

    ax.text(
        col + 0.5,
        row + 0.5,
        "T",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        zorder=6
    )

    # --------------------------------------------------------
    # ACTUAL TAXI-V4 WALLS
    # --------------------------------------------------------

    for wall in walls["vertical"]:

        row = wall["row"]
        col = wall["left_col"]

        ax.plot(
            [col + 1, col + 1],
            [row, row + 1],
            color="black",
            linewidth=6,
            solid_capstyle="butt",
            zorder=7
        )

    # Taxi-v4 currently has no internal horizontal walls.
    for wall in walls["horizontal"]:

        upper_row = wall["upper_row"]
        col = wall["col"]

        ax.plot(
            [col, col + 1],
            [upper_row + 1, upper_row + 1],
            color="black",
            linewidth=6,
            solid_capstyle="butt",
            zorder=7
        )

    # --------------------------------------------------------
    # AXIS LABELS
    # --------------------------------------------------------

    ax.set_xticks(
        [0.5, 1.5, 2.5, 3.5, 4.5]
    )

    ax.set_yticks(
        [0.5, 1.5, 2.5, 3.5, 4.5]
    )

    ax.set_xticklabels(
        ["0", "1", "2", "3", "4"]
    )

    ax.set_yticklabels(
        ["0", "1", "2", "3", "4"]
    )

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    ax.set_title(
        "Taxi-v4 Navigation Map",
        fontsize=16,
        fontweight="bold"
    )

    # --------------------------------------------------------
    # OUTER BORDER
    # --------------------------------------------------------

    for spine in ax.spines.values():

        spine.set_linewidth(2)

    plt.tight_layout()

    return fig


# ============================================================
# DISPLAY MAP
# ============================================================

walls = environment.get_walls()

figure = create_dashboard_map(
    taxi_position=current_taxi_position,
    passenger_position=passenger_position,
    destination_position=destination_position,
    walls=walls,
    route=route
)

st.subheader("🗺️ Taxi Environment")

st.pyplot(
    figure,
    use_container_width=True
)

plt.close(figure)


# ============================================================
# LEGEND
# ============================================================

st.markdown(
    """
    **Map Legend**

    `T` = Taxi &nbsp;&nbsp;
    `P` = Passenger &nbsp;&nbsp;
    `★` = Destination &nbsp;&nbsp;
    `R/G/Y/B` = Taxi-v4 landmarks &nbsp;&nbsp;
    **Black thick lines** = Actual Taxi-v4 movement barriers
    """
)


# ============================================================
# RESULTS
# ============================================================

if st.session_state.result is not None:

    result = st.session_state.result

    st.markdown("---")

    st.subheader("📊 Route Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "RL Agent Steps",
            result["rl_steps"]
        )

    with col2:

        st.metric(
            "Optimal Steps",
            result["optimal_steps"]
        )

    with col3:

        st.metric(
            "Extra Steps",
            result["extra_steps"]
        )

    with col4:

        st.metric(
            "Route Efficiency",
            f"{result['route_efficiency']:.1f}%"
        )

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    if result["rl_result"]["success"]:

        st.success(
            "✓ Q-Learning agent successfully completed "
            "the passenger pickup and destination dropoff."
        )

    else:

        st.error(
            "✗ Q-Learning agent did not complete the task."
        )

    # --------------------------------------------------------
    # OPTIMALITY
    # --------------------------------------------------------

    if result["is_optimal"]:

        st.success(
            "🏆 The learned route matches the shortest "
            "valid route for this task."
        )

    else:

        st.info(
            "The learned route completed the task but "
            "was longer than the optimal benchmark."
        )

    # --------------------------------------------------------
    # ACTION SEQUENCE
    # --------------------------------------------------------

    st.subheader("🧭 Learned Action Sequence")

    action_names = [
        environment.get_action_name(action)
        for action in result["rl_result"]["actions"]
    ]

    st.code(
        " → ".join(action_names)
    )

    # --------------------------------------------------------
    # ROUTE COORDINATES
    # --------------------------------------------------------

    st.subheader("📍 Route Coordinates")

    positions = result[
        "rl_result"
    ]["positions"]

    route_rows = []

    for index, position in enumerate(positions):

        route_rows.append(
            {
                "Step": index,
                "Row": position[0],
                "Column": position[1]
            }
        )

    st.dataframe(
        route_rows,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "Configure a taxi task from the sidebar and "
        "click **Start Navigation** to run the trained "
        "Q-Learning agent."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Taxi Route Optimization • "
    "Q-Learning + Taxi-v4 + BFS Optimal Benchmark"
)