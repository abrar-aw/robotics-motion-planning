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
        self.parent = None


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

    dist = distance(nearest, random_node)

    if dist <= step_size:

        return Node(
            random_node.x,
            random_node.y
        )

    theta = np.arctan2(
        random_node.y - nearest.y,
        random_node.x - nearest.x
    )

    new_x = nearest.x + step_size * np.cos(theta)
    new_y = nearest.y + step_size * np.sin(theta)

    return Node(new_x, new_y)


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

    # Sample points along the edge
    for t in np.linspace(0, 1, 20):

        x = node1.x + t * (node2.x - node1.x)
        y = node1.y + t * (node2.y - node1.y)

        for obstacle in OBSTACLES:

            if point_in_obstacle(x, y, obstacle):

                return False

    return True


# ============================================================
# RRT
# ============================================================

def rrt(
    max_iterations=5000,
    step_size=0.3,
    goal_tolerance=0.3
):

    start = Node(*START)
    goal = Node(*GOAL)

    tree = [start]

    for iteration in range(max_iterations):

        # 1. Random sample
        random = random_node()

        # 2. Find nearest tree node
        nearest = nearest_node(tree, random)

        # 3. Steer toward random sample
        new = steer(
            nearest,
            random,
            step_size
        )

        # 4. Check collision
        if not collision_free(nearest, new):
            continue

        # 5. Add new node
        new.parent = nearest
        tree.append(new)

        # 6. Check whether goal is reachable
        if distance(new, goal) <= goal_tolerance:

            if collision_free(new, goal):

                goal.parent = new
                tree.append(goal)

                return tree, goal, iteration + 1

    return tree, None, max_iterations


# ============================================================
# PATH EXTRACTION
# ============================================================

def get_path(goal):

    path = []

    current = goal

    while current is not None:

        path.append(
            (current.x, current.y)
        )

        current = current.parent

    path.reverse()

    return path


# ============================================================
# PATH LENGTH
# ============================================================

def path_length(path):

    length = 0.0

    for i in range(len(path) - 1):

        x1, y1 = path[i]
        x2, y2 = path[i + 1]

        length += np.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )

    return length


# ============================================================
# VISUALIZATION
# ============================================================

def plot_rrt(tree, path=None):

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    # Draw obstacles
    for x, y, width, height in OBSTACLES:

        rectangle = plt.Rectangle(
            (x, y),
            width,
            height
        )

        ax.add_patch(rectangle)

    # Draw tree
    for node in tree:

        if node.parent is not None:

            ax.plot(
                [node.x, node.parent.x],
                [node.y, node.parent.y],
                linewidth=0.5
            )

    # Start
    ax.scatter(
        START[0],
        START[1],
        s=100,
        marker="o",
        label="Start"
    )

    # Goal
    ax.scatter(
        GOAL[0],
        GOAL[1],
        s=150,
        marker="*",
        label="Goal"
    )

    # Final path
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
            label="RRT Path"
        )

    ax.set_xlim(X_LIMITS)
    ax.set_ylim(Y_LIMITS)

    ax.set_aspect("equal")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.set_title(
        "Rapidly-exploring Random Tree (RRT)"
    )

    ax.legend()

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Optional: makes the experiment reproducible
    np.random.seed(42)

    start_time = time.perf_counter()

    tree, goal_node, iterations = rrt()

    computation_time = (
        time.perf_counter() - start_time
    )

    if goal_node is not None:

        path = get_path(goal_node)

        length = path_length(path)

        print("\nRRT RESULT")
        print("-" * 30)
        print(f"Path found       : YES")
        print(f"Path length      : {length:.3f}")
        print(f"Computation time : {computation_time:.6f} s")
        print(f"Iterations       : {iterations}")
        print(f"Tree nodes       : {len(tree)}")

        plot_rrt(tree, path)

    else:

        print("\nRRT RESULT")
        print("-" * 30)
        print("Path found       : NO")
        print(f"Computation time : {computation_time:.6f} s")
        print(f"Iterations       : {iterations}")