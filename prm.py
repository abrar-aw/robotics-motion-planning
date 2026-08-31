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
# COLLISION CHECKING
# ============================================================

def point_in_obstacle(x, y, obstacle):

    ox, oy, width, height = obstacle

    return (
        ox <= x <= ox + width
        and
        oy <= y <= oy + height
    )


def collision_free(point1, point2):

    """
    Check whether the straight-line connection between
    two points is free of obstacles.
    """

    x1, y1 = point1
    x2, y2 = point2

    # Sample points along the edge
    for t in np.linspace(0, 1, 30):

        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        for obstacle in OBSTACLES:

            if point_in_obstacle(
                x,
                y,
                obstacle
            ):
                return False

    return True


# ============================================================
# DISTANCE
# ============================================================

def distance(point1, point2):

    return np.sqrt(
        (point1[0] - point2[0]) ** 2
        +
        (point1[1] - point2[1]) ** 2
    )


# ============================================================
# RANDOM FREE SAMPLES
# ============================================================

def generate_samples(num_samples):

    samples = []

    while len(samples) < num_samples:

        x = np.random.uniform(
            *X_LIMITS
        )

        y = np.random.uniform(
            *Y_LIMITS
        )

        # Only keep collision-free points
        valid = True

        for obstacle in OBSTACLES:

            if point_in_obstacle(
                x,
                y,
                obstacle
            ):

                valid = False
                break

        if valid:

            samples.append(
                (x, y)
            )

    return samples


# ============================================================
# BUILD ROADMAP
# ============================================================

def build_roadmap(
    samples,
    connection_radius=2.0
):

    graph = {
        i: []
        for i in range(len(samples))
    }

    for i in range(len(samples)):

        for j in range(
            i + 1,
            len(samples)
        ):

            dist = distance(
                samples[i],
                samples[j]
            )

            # Only consider nearby samples
            if dist <= connection_radius:

                # Check whether the edge is collision-free
                if collision_free(
                    samples[i],
                    samples[j]
                ):

                    graph[i].append(
                        (j, dist)
                    )

                    graph[j].append(
                        (i, dist)
                    )

    return graph


# ============================================================
# DIJKSTRA SHORTEST PATH
# ============================================================

def dijkstra(
    graph,
    start_index,
    goal_index
):

    distances = {
        node: float("inf")
        for node in graph
    }

    previous = {
        node: None
        for node in graph
    }

    distances[start_index] = 0.0

    priority_queue = [
        (0.0, start_index)
    ]

    while priority_queue:

        current_distance, current = (
            heapq.heappop(priority_queue)
        )

        # Ignore outdated queue entries
        if current_distance > distances[current]:
            continue

        # Goal reached
        if current == goal_index:
            break

        for neighbor, edge_cost in graph[current]:

            new_distance = (
                current_distance
                + edge_cost
            )

            if new_distance < distances[neighbor]:

                distances[neighbor] = new_distance

                previous[neighbor] = current

                heapq.heappush(
                    priority_queue,
                    (
                        new_distance,
                        neighbor
                    )
                )

    # No path exists
    if distances[goal_index] == float("inf"):

        return None, float("inf")

    # Reconstruct path
    path_indices = []

    current = goal_index

    while current is not None:

        path_indices.append(
            current
        )

        current = previous[current]

    path_indices.reverse()

    return (
        path_indices,
        distances[goal_index]
    )


# ============================================================
# CONNECT START AND GOAL
# ============================================================

def add_start_goal(
    samples,
    graph,
    start,
    goal,
    connection_radius=2.0
):

    # Start index
    start_index = len(samples)

    samples.append(start)

    graph[start_index] = []

    # Goal index
    goal_index = len(samples)

    samples.append(goal)

    graph[goal_index] = []

    # Connect start and goal to nearby roadmap nodes
    for i in range(
        len(samples) - 2
    ):

        # ----------------------------
        # Start connection
        # ----------------------------

        start_distance = distance(
            start,
            samples[i]
        )

        if (
            start_distance
            <= connection_radius
        ):

            if collision_free(
                start,
                samples[i]
            ):

                graph[start_index].append(
                    (
                        i,
                        start_distance
                    )
                )

                graph[i].append(
                    (
                        start_index,
                        start_distance
                    )
                )

        # ----------------------------
        # Goal connection
        # ----------------------------

        goal_distance = distance(
            goal,
            samples[i]
        )

        if (
            goal_distance
            <= connection_radius
        ):

            if collision_free(
                goal,
                samples[i]
            ):

                graph[goal_index].append(
                    (
                        i,
                        goal_distance
                    )
                )

                graph[i].append(
                    (
                        goal_index,
                        goal_distance
                    )
                )

    # Direct start → goal connection
    # if it is collision-free.
    direct_distance = distance(
        start,
        goal
    )

    if (
        direct_distance
        <= connection_radius
        and
        collision_free(
            start,
            goal
        )
    ):

        graph[start_index].append(
            (
                goal_index,
                direct_distance
            )
        )

        graph[goal_index].append(
            (
                start_index,
                direct_distance
            )
        )

    return (
        start_index,
        goal_index
    )


# ============================================================
# PRM
# ============================================================

def prm(
    num_samples=500,
    connection_radius=2.0
):

    # --------------------------------------------------------
    # Generate random samples
    # --------------------------------------------------------

    samples = generate_samples(
        num_samples
    )

    # --------------------------------------------------------
    # Build roadmap
    # --------------------------------------------------------

    graph = build_roadmap(
        samples,
        connection_radius
    )

    # --------------------------------------------------------
    # Add start and goal
    # --------------------------------------------------------

    start_index, goal_index = (
        add_start_goal(
            samples,
            graph,
            START,
            GOAL,
            connection_radius
        )
    )

    # --------------------------------------------------------
    # Find shortest path
    # --------------------------------------------------------

    path_indices, path_cost = dijkstra(
        graph,
        start_index,
        goal_index
    )

    # --------------------------------------------------------
    # Convert indices to coordinates
    # --------------------------------------------------------

    if path_indices is None:

        return (
            samples,
            graph,
            None,
            None
        )

    path = [
        samples[index]
        for index in path_indices
    ]

    return (
        samples,
        graph,
        path,
        path_cost
    )


# ============================================================
# PATH LENGTH
# ============================================================

def path_length(path):

    if path is None:
        return None

    length = 0.0

    for i in range(
        len(path) - 1
    ):

        length += distance(
            path[i],
            path[i + 1]
        )

    return length


# ============================================================
# VISUALIZATION
# ============================================================

def plot_prm(
    samples,
    graph,
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
    # Draw roadmap edges
    # --------------------------------------------------------

    drawn_edges = set()

    for node, neighbors in graph.items():

        for neighbor, _ in neighbors:

            edge = tuple(
                sorted(
                    (node, neighbor)
                )
            )

            if edge in drawn_edges:
                continue

            drawn_edges.add(edge)

            x1, y1 = samples[node]
            x2, y2 = samples[neighbor]

            ax.plot(
                [x1, x2],
                [y1, y2],
                linewidth=0.4,
                alpha=0.2
            )

    # --------------------------------------------------------
    # Draw roadmap nodes
    # --------------------------------------------------------

    sample_x = [
        point[0]
        for point in samples
    ]

    sample_y = [
        point[1]
        for point in samples
    ]

    ax.scatter(
        sample_x,
        sample_y,
        s=8,
        alpha=0.4
    )

    # --------------------------------------------------------
    # Draw final path
    # --------------------------------------------------------

    if path is not None:

        path_x = [
            point[0]
            for point in path
        ]

        path_y = [
            point[1]
            for point in path
        ]

        ax.plot(
            path_x,
            path_y,
            linewidth=3,
            label="PRM Path"
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
        "Probabilistic Roadmap (PRM)"
    )

    ax.legend()

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Reproducible experiment
    np.random.seed(42)

    # --------------------------------------------------------
    # Start timer
    # --------------------------------------------------------

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Run PRM
    # --------------------------------------------------------

    (
        samples,
        graph,
        path,
        path_cost
    ) = prm(
        num_samples=500,
        connection_radius=2.0
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
    print("PRM RESULT")
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
            f"Path cost        : "
            f"{path_cost:.3f}"
        )

        print(
            f"Computation time : "
            f"{computation_time:.6f} s"
        )

        print(
            f"Samples          : "
            f"{len(samples) - 2}"
        )

        print(
            f"Roadmap nodes    : "
            f"{len(graph)}"
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        plot_prm(
            samples,
            graph,
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
            f"Samples          : "
            f"{len(samples)}"
        )

        print(
            f"Roadmap nodes    : "
            f"{len(graph)}"
        )