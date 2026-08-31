import numpy as np
import matplotlib.pyplot as plt
import time


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
# NODE
# ============================================================

class Node:

    def __init__(self, x, y):

        self.x = x
        self.y = y

        # Parent node in the tree
        self.parent = None

        # Cost from start to this node
        self.cost = 0.0


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def distance(node1, node2):

    return np.sqrt(
        (node1.x - node2.x) ** 2 +
        (node1.y - node2.y) ** 2
    )


def random_node():

    x = np.random.uniform(*X_LIMITS)
    y = np.random.uniform(*Y_LIMITS)

    return Node(x, y)


def nearest_node(tree, random_node):

    return min(
        tree,
        key=lambda node: distance(node, random_node)
    )


# ============================================================
# STEERING
# ============================================================

def steer(nearest, random_node, step_size):

    dist = distance(
        nearest,
        random_node
    )

    # If random point is already close enough,
    # return a node at the random point.
    if dist <= step_size:

        return Node(
            random_node.x,
            random_node.y
        )

    # Direction from nearest node toward random node
    theta = np.arctan2(
        random_node.y - nearest.y,
        random_node.x - nearest.x
    )

    new_x = (
        nearest.x
        + step_size * np.cos(theta)
    )

    new_y = (
        nearest.y
        + step_size * np.sin(theta)
    )

    return Node(
        new_x,
        new_y
    )


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


def collision_free(node1, node2):

    # Sample points along the edge between node1 and node2
    for t in np.linspace(0, 1, 20):

        x = (
            node1.x
            + t * (node2.x - node1.x)
        )

        y = (
            node1.y
            + t * (node2.y - node1.y)
        )

        for obstacle in OBSTACLES:

            if point_in_obstacle(
                x,
                y,
                obstacle
            ):

                return False

    return True


# ============================================================
# FIND NEARBY NODES
# ============================================================

def near_nodes(tree, new_node, radius):

    neighbors = []

    for node in tree:

        if distance(node, new_node) <= radius:

            neighbors.append(node)

    return neighbors


# ============================================================
# CHOOSE BEST PARENT
# ============================================================

def choose_parent(neighbors, new_node):

    best_parent = None
    best_cost = float("inf")

    for node in neighbors:

        # Make sure the edge is collision-free
        if collision_free(
            node,
            new_node
        ):

            # Cost of reaching new_node through node
            new_cost = (
                node.cost
                + distance(node, new_node)
            )

            if new_cost < best_cost:

                best_cost = new_cost
                best_parent = node

    return best_parent, best_cost


# ============================================================
# UPDATE DESCENDANT COSTS
# ============================================================

def update_descendant_costs(node, tree):

    """
    When a node is rewired, its descendants may also
    have a cheaper path to the start.

    Recursively update their costs.
    """

    for child in tree:

        if child.parent is node:

            child.cost = (
                node.cost
                + distance(node, child)
            )

            update_descendant_costs(
                child,
                tree
            )


# ============================================================
# REWIRE
# ============================================================

def rewire(new_node, neighbors, tree):

    """
    Check whether nearby nodes can obtain a cheaper
    path by using new_node as their parent.
    """

    for node in neighbors:

        new_cost = (
            new_node.cost
            + distance(new_node, node)
        )

        # Only rewire if the new route is cheaper
        if new_cost < node.cost:

            # Make sure the new connection is collision-free
            if collision_free(
                new_node,
                node
            ):

                node.parent = new_node
                node.cost = new_cost

                # Update all descendants
                update_descendant_costs(
                    node,
                    tree
                )


# ============================================================
# RRT*
# ============================================================

def rrt_star(
    max_iterations=5000,
    step_size=0.3,
    goal_tolerance=0.3,
    search_radius=1.0
):

    start = Node(
        *START
    )

    goal = Node(
        *GOAL
    )

    # Initial tree contains only the start node
    tree = [start]

    # Best connection to goal found so far
    best_goal = None

    for iteration in range(
        max_iterations
    ):

        # ----------------------------------------------------
        # 1. Random sampling
        # ----------------------------------------------------

        random = random_node()

        # ----------------------------------------------------
        # 2. Find nearest existing node
        # ----------------------------------------------------

        nearest = nearest_node(
            tree,
            random
        )

        # ----------------------------------------------------
        # 3. Steer toward random sample
        # ----------------------------------------------------

        new = steer(
            nearest,
            random,
            step_size
        )

        # ----------------------------------------------------
        # 4. Check collision
        # ----------------------------------------------------

        if not collision_free(
            nearest,
            new
        ):

            continue

        # ----------------------------------------------------
        # 5. Find nearby nodes
        # ----------------------------------------------------

        neighbors = near_nodes(
            tree,
            new,
            search_radius
        )

        # ----------------------------------------------------
        # 6. Choose cheapest parent
        # ----------------------------------------------------

        best_parent, best_cost = choose_parent(
            neighbors,
            new
        )

        # If no valid nearby parent exists,
        # fall back to nearest node.

        if best_parent is None:

            best_parent = nearest

            best_cost = (
                nearest.cost
                + distance(
                    nearest,
                    new
                )
            )

        # ----------------------------------------------------
        # 7. Add node to tree
        # ----------------------------------------------------

        new.parent = best_parent
        new.cost = best_cost

        tree.append(new)

        # ----------------------------------------------------
        # 8. Rewire nearby nodes
        # ----------------------------------------------------

        rewire(
            new,
            neighbors,
            tree
        )

        # ----------------------------------------------------
        # 9. Check connection to goal
        # ----------------------------------------------------

        if distance(
            new,
            goal
        ) <= goal_tolerance:

            if collision_free(
                new,
                goal
            ):

                candidate_cost = (
                    new.cost
                    + distance(
                        new,
                        goal
                    )
                )

                # Keep the best goal connection
                if (
                    best_goal is None
                    or candidate_cost
                    < best_goal.cost
                ):

                    best_goal = Node(
                        goal.x,
                        goal.y
                    )

                    best_goal.parent = new
                    best_goal.cost = candidate_cost

    return (
        tree,
        best_goal,
        max_iterations
    )


# ============================================================
# PATH EXTRACTION
# ============================================================

def get_path(goal):

    path = []

    current = goal

    while current is not None:

        path.append(
            (
                current.x,
                current.y
            )
        )

        current = current.parent

    path.reverse()

    return path


# ============================================================
# PATH LENGTH
# ============================================================

def path_length(path):

    length = 0.0

    for i in range(
        len(path) - 1
    ):

        x1, y1 = path[i]
        x2, y2 = path[i + 1]

        length += np.sqrt(
            (x2 - x1) ** 2
            + (y2 - y1) ** 2
        )

    return length


# ============================================================
# VISUALIZATION
# ============================================================

def plot_rrt_star(
    tree,
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
    # Draw RRT* tree
    # --------------------------------------------------------

    for node in tree:

        if node.parent is not None:

            ax.plot(
                [
                    node.x,
                    node.parent.x
                ],
                [
                    node.y,
                    node.parent.y
                ],
                linewidth=0.4,
                alpha=0.25
            )

    # --------------------------------------------------------
    # Draw start
    # --------------------------------------------------------

    ax.scatter(
        START[0],
        START[1],
        s=100,
        marker="o",
        label="Start"
    )

    # --------------------------------------------------------
    # Draw goal
    # --------------------------------------------------------

    ax.scatter(
        GOAL[0],
        GOAL[1],
        s=150,
        marker="*",
        label="Goal"
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
            label="RRT* Path"
        )

    # --------------------------------------------------------
    # Plot formatting
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
        "RRT* Motion Planning"
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
    # Run RRT*
    # --------------------------------------------------------

    tree, goal_node, iterations = rrt_star()

    # --------------------------------------------------------
    # Stop timer
    # --------------------------------------------------------

    computation_time = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------
    # Process result
    # --------------------------------------------------------

    if goal_node is not None:

        path = get_path(
            goal_node
        )

        length = path_length(
            path
        )

        print()
        print("RRT* RESULT")
        print("-" * 30)

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
            f"Iterations       : "
            f"{iterations}"
        )

        print(
            f"Tree nodes       : "
            f"{len(tree)}"
        )

        print(
            f"Path cost        : "
            f"{length:.3f}"
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        plot_rrt_star(
            tree,
            path
        )

    else:

        print()
        print("RRT* RESULT")
        print("-" * 30)

        print(
            "Path found       : NO"
        )

        print(
            f"Computation time : "
            f"{computation_time:.6f} s"
        )

        print(
            f"Iterations       : "
            f"{iterations}"
        )

        print(
            f"Tree nodes       : "
            f"{len(tree)}"
        )