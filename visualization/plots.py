import matplotlib.pyplot as plt


def create_taxi_map(
    taxi_position=None,
    passenger_position=None,
    destination_position=None,
    route=None,
    walls=None,
    title="Taxi Route Optimization"
):

    fig, ax = plt.subplots(figsize=(8, 8))

    # ========================================================
    # MAP SETTINGS
    # ========================================================

    ax.set_xlim(0, 5)
    ax.set_ylim(5, 0)
    ax.set_aspect("equal")

    ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5])
    ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5])

    ax.set_xticklabels(["0", "1", "2", "3", "4"])
    ax.set_yticklabels(["0", "1", "2", "3", "4"])

    ax.set_xlabel("Column", fontsize=11)
    ax.set_ylabel("Row", fontsize=11)

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    # ========================================================
    # LIGHT GRID
    # ========================================================

    # Normal cell boundaries are intentionally faint.
    # Actual Taxi-v4 walls are drawn separately below.

    for x in range(6):
        ax.plot(
            [x, x],
            [0, 5],
            linewidth=0.8,
            alpha=0.20,
            zorder=1
        )

    for y in range(6):
        ax.plot(
            [0, 5],
            [y, y],
            linewidth=0.8,
            alpha=0.20,
            zorder=1
        )

    # ========================================================
    # TAXI-V4 LANDMARKS
    # ========================================================

    landmarks = {
        0: ("R", "Red", (0, 0)),
        1: ("G", "Green", (0, 4)),
        2: ("Y", "Yellow", (4, 0)),
        3: ("B", "Blue", (4, 3))
    }

    for _, (symbol, name, position) in landmarks.items():

        row, col = position

        ax.text(
            col + 0.12,
            row + 0.22,
            symbol,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="top",
            zorder=2
        )

        ax.text(
            col + 0.12,
            row + 0.82,
            name,
            fontsize=7,
            ha="left",
            va="bottom",
            alpha=0.65,
            zorder=2
        )

    # ========================================================
    # ROUTE
    # ========================================================

    if route:

        route_positions = [
            (col + 0.5, row + 0.5)
            for row, col in route
        ]

        x_values = [
            position[0]
            for position in route_positions
        ]

        y_values = [
            position[1]
            for position in route_positions
        ]

        ax.plot(
            x_values,
            y_values,
            linewidth=3,
            marker="o",
            markersize=5,
            alpha=0.65,
            zorder=3
        )

    # ========================================================
    # ACTUAL TAXI-V4 WALLS
    # ========================================================
    #
    # Walls are boundaries BETWEEN cells.
    #
    # Vertical wall:
    #
    #       Cell 0 | Cell 1
    #              ↑
    #             WALL
    #
    # Horizontal wall:
    #
    #       Cell
    #       -----
    #        WALL
    #       -----
    #       Cell
    #
    # They are drawn AFTER the route so that the barriers
    # remain visually clear.
    # ========================================================

    if walls:

        # ----------------------------------------------------
        # Vertical walls
        # ----------------------------------------------------

        for wall in walls.get("vertical", []):

            row = wall["row"]
            left_col = wall["left_col"]

            x = left_col + 1

            ax.plot(
                [x, x],
                [row, row + 1],
                linewidth=7,
                solid_capstyle="butt",
                zorder=6
            )

        # ----------------------------------------------------
        # Horizontal walls
        # ----------------------------------------------------

        for wall in walls.get("horizontal", []):

            upper_row = wall["upper_row"]
            col = wall["col"]

            y = upper_row + 1

            ax.plot(
                [col, col + 1],
                [y, y],
                linewidth=7,
                solid_capstyle="butt",
                zorder=6
            )

    # ========================================================
    # PASSENGER
    # ========================================================

    if passenger_position is not None:

        row, col = passenger_position

        ax.scatter(
            col + 0.5,
            row + 0.5,
            s=360,
            marker="o",
            facecolors="none",
            linewidths=2,
            zorder=7
        )

        ax.text(
            col + 0.5,
            row + 0.5,
            "P",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            zorder=8
        )

    # ========================================================
    # DESTINATION
    # ========================================================

    if destination_position is not None:

        row, col = destination_position

        ax.scatter(
            col + 0.5,
            row + 0.5,
            s=450,
            marker="*",
            linewidths=1.5,
            zorder=7
        )

        ax.text(
            col + 0.5,
            row + 0.82,
            "DEST",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            zorder=8
        )

    # ========================================================
    # TAXI
    # ========================================================

    if taxi_position is not None:

        row, col = taxi_position

        ax.scatter(
            col + 0.5,
            row + 0.5,
            s=500,
            marker="s",
            linewidths=2,
            zorder=9
        )

        ax.text(
            col + 0.5,
            row + 0.5,
            "T",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            zorder=10
        )

    # ========================================================
    # MAP BORDER
    # ========================================================

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # ========================================================
    # LEGEND
    # ========================================================

    ax.text(
        0,
        5.25,
        "T = Taxi    P = Passenger    ★ = Destination    "
        "Thick barriers = Walls",
        fontsize=9
    )

    # ========================================================
    # FINAL LAYOUT
    # ========================================================

    plt.tight_layout()

    return fig