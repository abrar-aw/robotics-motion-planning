import numpy as np
import matplotlib.pyplot as plt
import time
import heapq


# ============================================================
# ENVIRONMENT
# ============================================================

X_LIMITS = (0, 10)
Y_LIMITS = (0, 10)

START = (1, 1)
GOAL = (9, 9)

OBSTACLES = [
    (3, 2, 2, 5),
    (6, 6, 2, 3),
    (4, 7, 1, 1)
]


# ============================================================
# GRID SETTINGS
# ============================================================

# Number of cells along each axis.
# Higher resolution = more accurate path,
# but potentially more computation.
GRID_SIZE = 100


# ============================================================
# OBSTACLE CHECKING
# ============================================================

def point_in_obstacle(x, y, obstacle):

    ox, oy, width, height = obstacle

    return (
        ox <= x <= ox + width
        and
        oy <= y <= oy + height
    )


def point_is_free(x, y):

    for obstacle in OBSTACLES:

        if point_in_obstacle(
            x,
            y,
            obstacle
        ):
            return False

    return True


# ============================================================
# COORDINATE CONVERSION
# ============================================================

def world_to_grid(point):

    x, y = point

    gx = int(
        round(
            x / X_LIMITS[1]
            * (GRID_SIZE - 1)
        )
    )

    gy = int(
        round(
            y / Y_LIMITS[1]
            * (GRID_SIZE - 1)
        )
    )

    gx = np.clip(
        gx,
        0,
        GRID_SIZE - 1
    )

    gy = np.clip(
        gy,
        0,
        GRID_SIZE - 1
    )

    return (
        int(gx),
        int(gy)
    )


def grid_to_world(cell):

    gx, gy = cell

    x = (
        gx
        / (GRID_SIZE - 1)
        * X_LIMITS[1]
    )

    y = (
        gy
        / (GRID_SIZE - 1)
        * Y_LIMITS[1]
    )

    return (
        x,
        y
    )


# ============================================================
# CREATE OCCUPANCY GRID
# ============================================================

def create_grid():

    grid = np.zeros(
        (
            GRID_SIZE,
            GRID_SIZE
        ),
        dtype=bool
    )

    for gx in range(GRID_SIZE):

        for gy in range(GRID_SIZE):

            x, y = grid_to_world(
                (gx, gy)
            )

            if not point_is_free(x, y):

                grid[gy, gx] = True

    return grid


# ============================================================
# HEURISTIC
# ============================================================

def heuristic(node, goal):

    """
    Euclidean distance heuristic.
    """

    dx = node[0] - goal[0]
    dy = node[1] - goal[1]

    return np.sqrt(
        dx ** 2
        + dy ** 2
    )


# ============================================================
# A* SEARCH
# ============================================================

def astar(grid, start, goal):

    # --------------------------------------------------------
    # Movement directions
    #
    # 8-connected grid:
    #
    # NW  N  NE
    #  W  X   E
    # SW  S  SE
    # --------------------------------------------------------

    neighbors = [
        (-1, -1),
        (-1,  0),
        (-1,  1),
        ( 0, -1),
        ( 0,  1),
        ( 1, -1),
        ( 1,  0),
        ( 1,  1)
    ]

    # Cost from start
    g_cost = {
        start: 0.0
    }

    # Parent of each node
    came_from = {}

    # Priority queue
    open_set = []

    start_priority = heuristic(
        start,
        goal
    )

    heapq.heappush(
        open_set,
        (
            start_priority,
            start
        )
    )

    # Nodes that have already been explored
    closed_set = set()

    nodes_explored = 0

    # --------------------------------------------------------
    # Main A* loop
    # --------------------------------------------------------

    while open_set:

        current_priority, current = (
            heapq.heappop(open_set)
        )

        # Ignore nodes already processed
        if current in closed_set:
            continue

        closed_set.add(current)

        nodes_explored += 1

        # ----------------------------------------------------
        # Goal reached
        # ----------------------------------------------------

        if current == goal:

            path = []

            node = current

            while node != start:

                path.append(node)

                node = came_from[node]

            path.append(start)

            path.reverse()

            return (
                path,
                g_cost[current],
                nodes_explored
            )

        # ----------------------------------------------------
        # Explore neighbors
        # ----------------------------------------------------

        for dx, dy in neighbors:

            nx = current[0] + dx
            ny = current[1] + dy

            neighbor = (
                nx,
                ny
            )

            # Check grid boundaries
            if (
                nx < 0
                or nx >= GRID_SIZE
                or ny < 0
                or ny >= GRID_SIZE
            ):
                continue

            # Check obstacle
            if grid[ny, nx]:

                continue

            # Prevent diagonal movement through
            # the corner of two obstacles.
            if dx != 0 and dy != 0:

                side1 = (
                    current[0] + dx,
                    current[1]
                )

                side2 = (
                    current[0],
                    current[1] + dy
                )

                if (
                    grid[
                        side1[1],
                        side1[0]
                    ]
                    or
                    grid[
                        side2[1],
                        side2[0]
                    ]
                ):

                    continue

            # Straight movement = 1
            # Diagonal movement = sqrt(2)
            if dx != 0 and dy != 0:

                movement_cost = np.sqrt(2)

            else:

                movement_cost = 1.0

            tentative_g = (
                g_cost[current]
                + movement_cost
            )

            # If this is a better path
            if (
                neighbor not in g_cost
                or
                tentative_g
                < g_cost[neighbor]
            ):

                g_cost[neighbor] = (
                    tentative_g
                )

                came_from[neighbor] = (
                    current
                )

                f_cost = (
                    tentative_g
                    + heuristic(
                        neighbor,
                        goal
                    )
                )

                heapq.heappush(
                    open_set,
                    (
                        f_cost,
                        neighbor
                    )
                )

    # No path found
    return (
        None,
        float("inf"),
        nodes_explored
    )


# ============================================================
# PATH LENGTH IN WORLD COORDINATES
# ============================================================

def path_length(path):

    if path is None:

        return None

    length = 0.0

    for i in range(
        len(path) - 1
    ):

        x1, y1 = grid_to_world(
            path[i]
        )

        x2, y2 = grid_to_world(
            path[i + 1]
        )

        length += np.sqrt(
            (x2 - x1) ** 2
            +
            (y2 - y1) ** 2
        )

    return length


# ============================================================
# VISUALIZATION
# ============================================================

def plot_astar(
    grid,
    path=None
):

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    # --------------------------------------------------------
    # Draw obstacles
    # --------------------------------------------------------

    for (
        x,
        y,
        width,
        height
    ) in OBSTACLES:

        rectangle = plt.Rectangle(
            (x, y),
            width,
            height
        )

        ax.add_patch(
            rectangle
        )

    # --------------------------------------------------------
    # Draw explored/free grid lightly
    # --------------------------------------------------------

    ax.imshow(
        grid,
        origin="lower",
        extent=[
            X_LIMITS[0],
            X_LIMITS[1],
            Y_LIMITS[0],
            Y_LIMITS[1]
        ],
        interpolation="nearest",
        alpha=0.12
    )

    # --------------------------------------------------------
    # Draw final path
    # --------------------------------------------------------

    if path is not None:

        path_world = [
            grid_to_world(cell)
            for cell in path
        ]

        path_x = [
            point[0]
            for point in path_world
        ]

        path_y = [
            point[1]
            for point in path_world
        ]

        ax.plot(
            path_x,
            path_y,
            linewidth=3,
            label="A* Path"
        )

    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------

    ax.scatter(
        START[0],
        START[1],
        s=100,
        marker="o",
        label="Start"
    )

    # --------------------------------------------------------
    # Goal
    # --------------------------------------------------------

    ax.scatter(
        GOAL[0],
        GOAL[1],
        s=150,
        marker="*",
        label="Goal"
    )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_xlim(
        X_LIMITS
    )

    ax.set_ylim(
        Y_LIMITS
    )

    ax.set_aspect(
        "equal"
    )

    ax.set_xlabel(
        "X"
    )

    ax.set_ylabel(
        "Y"
    )

    ax.set_title(
        "A* Grid-Based Motion Planning"
    )

    ax.legend()

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Create occupancy grid
    # --------------------------------------------------------

    grid = create_grid()

    # --------------------------------------------------------
    # Convert start and goal to grid coordinates
    # --------------------------------------------------------

    start_grid = world_to_grid(
        START
    )

    goal_grid = world_to_grid(
        GOAL
    )

    # --------------------------------------------------------
    # Start timer
    # --------------------------------------------------------

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Run A*
    # --------------------------------------------------------

    (
        path,
        grid_cost,
        nodes_explored
    ) = astar(
        grid,
        start_grid,
        goal_grid
    )

    # --------------------------------------------------------
    # Stop timer
    # --------------------------------------------------------

    computation_time = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("A* RESULT")
    print("-" * 30)

    if path is not None:

        length = path_length(
            path
        )

        print(
            "Path found       : YES"
        )

        print(
            f"Path length      : "
            f"{length:.3f}"
        )

        print(
            f"Computation time : "
            f"{computation_time:.6f} s"
        )

        print(
            f"Nodes explored   : "
            f"{nodes_explored}"
        )

        print(
            f"Grid resolution  : "
            f"{GRID_SIZE} x {GRID_SIZE}"
        )

        print(
            f"Path nodes       : "
            f"{len(path)}"
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        plot_astar(
            grid,
            path
        )

    else:

        print(
            "Path found       : NO"
        )

        print(
            f"Computation time : "
            f"{computation_time:.6f} s"
        )

        print(
            f"Nodes explored   : "
            f"{nodes_explored}"
        )

        print(
            f"Grid resolution  : "
            f"{GRID_SIZE} x {GRID_SIZE}"
        )