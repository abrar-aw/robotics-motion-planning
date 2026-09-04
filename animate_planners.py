import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ============================================================
# PROJECT IMPORTS
# ============================================================

from environment import (
    X_LIMITS,
    Y_LIMITS,
    START,
    GOAL,
    OBSTACLES
)

from rrt import rrt, get_path
from rrt_star import rrt_star
from prm import prm
from astar import (
    astar,
    create_grid,
    world_to_grid,
    grid_to_world
)


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

FIGURES_DIR = os.path.join(
    BASE_DIR,
    "figures"
)

os.makedirs(
    FIGURES_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# Animation speed
# ------------------------------------------------------------

FPS = 15

RRT_SECONDS = 9
RRT_STAR_SECONDS = 10
PRM_SECONDS = 9
ASTAR_SECONDS = 8


# ============================================================
# COMMON PLOT SETUP
# ============================================================

def setup_plot(title):

    # Smaller figure with tighter margins
    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    # --------------------------------------------------------
    # Obstacles
    # --------------------------------------------------------

    for x, y, width, height in OBSTACLES:

        rectangle = plt.Rectangle(
            (x, y),
            width,
            height,
            alpha=0.8
        )

        ax.add_patch(
            rectangle
        )

    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------

    ax.scatter(
        START[0],
        START[1],
        s=100,
        marker="o",
        label="Start",
        zorder=10
    )

    # --------------------------------------------------------
    # Goal
    # --------------------------------------------------------

    ax.scatter(
        GOAL[0],
        GOAL[1],
        s=150,
        marker="*",
        label="Goal",
        zorder=10
    )

    # --------------------------------------------------------
    # Axes
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
        title
    )

    # --------------------------------------------------------
    # Tight margins
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.08,
        top=0.92
    )

    return fig, ax


# ============================================================
# SAVE GIF
# ============================================================

def save_gif(
    animation,
    filename
):

    writer = PillowWriter(
        fps=FPS
    )

    animation.save(
        filename,
        writer=writer
    )

    print(
        f"Saved: {filename}"
    )


# ============================================================
# RRT ANIMATION
# ============================================================

def animate_rrt():

    print(
        "\nGenerating RRT animation..."
    )

    # --------------------------------------------------------
    # Reproducible run
    # --------------------------------------------------------

    np.random.seed(42)

    tree, goal_node, iterations = rrt()

    if goal_node is not None:

        path = get_path(
            goal_node
        )

    else:

        path = None

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, ax = setup_plot(
        "RRT - Tree Construction"
    )

    # --------------------------------------------------------
    # Extract tree edges
    # --------------------------------------------------------

    edges = []

    for node in tree:

        if node.parent is not None:

            edges.append(
                (
                    node.x,
                    node.y,
                    node.parent.x,
                    node.parent.y
                )
            )

    # --------------------------------------------------------
    # Animation timing
    # --------------------------------------------------------

    total_frames = (
        RRT_SECONDS * FPS
    )

    # Tree construction takes ~75%
    construction_frames = int(
        total_frames * 0.75
    )

    # --------------------------------------------------------
    # Display edges
    # --------------------------------------------------------

    max_display_edges = 1800

    display_count = min(
        len(edges),
        max_display_edges
    )

    # Compress the complete tree into the available frames

    edge_indices = np.linspace(
        0,
        display_count,
        construction_frames
    ).astype(int)

    tree_lines = []

    for _ in range(
        display_count
    ):

        line, = ax.plot(
            [],
            [],
            linewidth=0.55,
            alpha=0.45
        )

        tree_lines.append(
            line
        )

    # --------------------------------------------------------
    # Final path
    # --------------------------------------------------------

    path_line, = ax.plot(
        [],
        [],
        linewidth=3,
        label="RRT Path",
        zorder=8
    )

    # --------------------------------------------------------
    # Animation update
    # --------------------------------------------------------

    def update(frame):

        # ====================================================
        # TREE CONSTRUCTION
        # ====================================================

        if frame < construction_frames:

            target_edges = edge_indices[
                frame
            ]

            target_edges = min(
                target_edges,
                display_count
            )

            for i in range(
                target_edges
            ):

                x1, y1, x2, y2 = edges[i]

                tree_lines[i].set_data(
                    [x1, x2],
                    [y1, y2]
                )

        # ====================================================
        # FINAL PATH
        # ====================================================

        else:

            if path is not None:

                path_x = [
                    point[0]
                    for point in path
                ]

                path_y = [
                    point[1]
                    for point in path
                ]

                path_line.set_data(
                    path_x,
                    path_y
                )

        return (
            tree_lines
            + [path_line]
        )

    # --------------------------------------------------------
    # Create animation
    # --------------------------------------------------------

    animation = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=1000 / FPS,
        blit=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    filename = os.path.join(
        FIGURES_DIR,
        "rrt_animation.gif"
    )

    save_gif(
        animation,
        filename
    )

    plt.close(
        fig
    )


# ============================================================
# RRT* ANIMATION
# ============================================================

def animate_rrt_star():

    print(
        "\nGenerating RRT* animation..."
    )

    # --------------------------------------------------------
    # Reproducible run
    # --------------------------------------------------------

    np.random.seed(42)

    result = rrt_star()

    # Your RRT* returns tree, goal_node, iterations

    if len(result) == 3:

        tree, goal_node, iterations = result

    else:

        tree, goal_node = result
        iterations = len(tree)

    if goal_node is not None:

        path = get_path(
            goal_node
        )

    else:

        path = None

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, ax = setup_plot(
        "RRT* - Tree Construction and Optimization"
    )

    # --------------------------------------------------------
    # Extract tree edges
    # --------------------------------------------------------

    edges = []

    for node in tree:

        if node.parent is not None:

            edges.append(
                (
                    node.x,
                    node.y,
                    node.parent.x,
                    node.parent.y
                )
            )

    # --------------------------------------------------------
    # Animation timing
    # --------------------------------------------------------

    total_frames = (
        RRT_STAR_SECONDS * FPS
    )

    construction_frames = int(
        total_frames * 0.70
    )

    # --------------------------------------------------------
    # Display edges
    # --------------------------------------------------------

    max_display_edges = 1800

    display_count = min(
        len(edges),
        max_display_edges
    )

    edge_indices = np.linspace(
        0,
        display_count,
        construction_frames
    ).astype(int)

    tree_lines = []

    for _ in range(
        display_count
    ):

        line, = ax.plot(
            [],
            [],
            linewidth=0.55,
            alpha=0.4
        )

        tree_lines.append(
            line
        )

    # --------------------------------------------------------
    # Final path
    # --------------------------------------------------------

    path_line, = ax.plot(
        [],
        [],
        linewidth=3,
        label="RRT* Path",
        zorder=8
    )

    # --------------------------------------------------------
    # Animation update
    # --------------------------------------------------------

    def update(frame):

        # ====================================================
        # TREE CONSTRUCTION
        # ====================================================

        if frame < construction_frames:

            target_edges = edge_indices[
                frame
            ]

            target_edges = min(
                target_edges,
                display_count
            )

            for i in range(
                target_edges
            ):

                x1, y1, x2, y2 = edges[i]

                tree_lines[i].set_data(
                    [x1, x2],
                    [y1, y2]
                )

        # ====================================================
        # FINAL PATH
        # ====================================================

        else:

            if path is not None:

                path_x = [
                    point[0]
                    for point in path
                ]

                path_y = [
                    point[1]
                    for point in path
                ]

                path_line.set_data(
                    path_x,
                    path_y
                )

        return (
            tree_lines
            + [path_line]
        )

    # --------------------------------------------------------
    # Create animation
    # --------------------------------------------------------

    animation = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=1000 / FPS,
        blit=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    filename = os.path.join(
        FIGURES_DIR,
        "rrt_star_animation.gif"
    )

    save_gif(
        animation,
        filename
    )

    plt.close(
        fig
    )


# ============================================================
# PRM ANIMATION
# ============================================================

def animate_prm():

    print(
        "\nGenerating PRM animation..."
    )

    # --------------------------------------------------------
    # Reproducible run
    # --------------------------------------------------------

    np.random.seed(42)

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
    # Figure
    # --------------------------------------------------------

    fig, ax = setup_plot(
        "PRM - Roadmap Construction"
    )

    # --------------------------------------------------------
    # Extract unique roadmap edges
    #
    # graph format:
    #
    # graph[node] = [
    #     (neighbor, distance),
    #     ...
    # ]
    # --------------------------------------------------------

    edges = []

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

            drawn_edges.add(
                edge
            )

            x1, y1 = samples[node]
            x2, y2 = samples[neighbor]

            edges.append(
                (
                    x1,
                    y1,
                    x2,
                    y2
                )
            )

    print(
        f"PRM samples: {len(samples) - 2}"
    )

    print(
        f"PRM roadmap edges: {len(edges)}"
    )

    # --------------------------------------------------------
    # Sample coordinates
    # --------------------------------------------------------

    sample_x = [
        point[0]
        for point in samples
    ]

    sample_y = [
        point[1]
        for point in samples
    ]

    sample_plot, = ax.plot(
        [],
        [],
        "o",
        markersize=2.5,
        alpha=0.55,
        linestyle="None"
    )

    # --------------------------------------------------------
    # Roadmap lines
    # --------------------------------------------------------

    roadmap_lines = []

    for _ in edges:

        line, = ax.plot(
            [],
            [],
            linewidth=0.35,
            alpha=0.25
        )

        roadmap_lines.append(
            line
        )

    # --------------------------------------------------------
    # Final path
    # --------------------------------------------------------

    path_line, = ax.plot(
        [],
        [],
        linewidth=3,
        label="PRM Path",
        zorder=8
    )

    # --------------------------------------------------------
    # Animation timing
    #
    # 20% -> samples
    # 55% -> roadmap
    # 25% -> final path / pause
    # --------------------------------------------------------

    total_frames = (
        PRM_SECONDS * FPS
    )

    sample_frames = int(
        total_frames * 0.20
    )

    roadmap_frames = int(
        total_frames * 0.55
    )

    path_start = (
        sample_frames
        + roadmap_frames
    )

    # --------------------------------------------------------
    # Animation update
    # --------------------------------------------------------

    def update(frame):

        # ====================================================
        # PHASE 1: SAMPLE FREE SPACE
        # ====================================================

        if frame < sample_frames:

            progress = (
                (frame + 1)
                / sample_frames
            )

            count = int(
                len(samples)
                * progress
            )

            sample_plot.set_data(
                sample_x[:count],
                sample_y[:count]
            )

        else:

            sample_plot.set_data(
                sample_x,
                sample_y
            )

        # ====================================================
        # PHASE 2: BUILD ROADMAP
        # ====================================================

        if frame >= sample_frames:

            roadmap_progress = (
                frame
                - sample_frames
            )

            progress = min(
                roadmap_progress
                / roadmap_frames,
                1.0
            )

            count = int(
                len(edges)
                * progress
            )

            for i in range(
                count
            ):

                x1, y1, x2, y2 = edges[i]

                roadmap_lines[i].set_data(
                    [x1, x2],
                    [y1, y2]
                )

        # ====================================================
        # PHASE 3: FINAL PATH
        # ====================================================

        if (
            frame >= path_start
            and path is not None
        ):

            path_x = [
                point[0]
                for point in path
            ]

            path_y = [
                point[1]
                for point in path
            ]

            path_line.set_data(
                path_x,
                path_y
            )

        return (
            [sample_plot, path_line]
            + roadmap_lines
        )

    # --------------------------------------------------------
    # Create animation
    # --------------------------------------------------------

    animation = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=1000 / FPS,
        blit=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    filename = os.path.join(
        FIGURES_DIR,
        "prm_animation.gif"
    )

    save_gif(
        animation,
        filename
    )

    plt.close(
        fig
    )


# ============================================================
# A* ANIMATION
# ============================================================

def animate_astar():

    print(
        "\nGenerating A* animation..."
    )

    # --------------------------------------------------------
    # Create occupancy grid
    # --------------------------------------------------------

    grid = create_grid()

    start_grid = world_to_grid(
        START
    )

    goal_grid = world_to_grid(
        GOAL
    )

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

    if path is None:

        print(
            "A* did not find a path."
        )

        return

    # --------------------------------------------------------
    # Convert grid path to world coordinates
    # --------------------------------------------------------

    path_world = [
        grid_to_world(cell)
        for cell in path
    ]

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, ax = setup_plot(
        "A* - Grid-Based Path Search"
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.set_xticks(
        np.arange(
            X_LIMITS[0],
            X_LIMITS[1] + 1,
            1
        ),
        minor=True
    )

    ax.set_yticks(
        np.arange(
            Y_LIMITS[0],
            Y_LIMITS[1] + 1,
            1
        ),
        minor=True
    )

    ax.grid(
        which="minor",
        linewidth=0.2,
        alpha=0.3
    )

    # --------------------------------------------------------
    # Final path
    # --------------------------------------------------------

    path_line, = ax.plot(
        [],
        [],
        linewidth=3,
        label="A* Path",
        zorder=8
    )

    # --------------------------------------------------------
    # Animation timing
    # --------------------------------------------------------

    total_frames = (
        ASTAR_SECONDS * FPS
    )

    path_frames = int(
        total_frames * 0.65
    )

    # --------------------------------------------------------
    # Animation update
    # --------------------------------------------------------

    def update(frame):

        # ----------------------------------------------------
        # Gradually reveal final path
        # ----------------------------------------------------

        if frame < path_frames:

            progress = (
                (frame + 1)
                / path_frames
            )

            count = max(
                2,
                int(
                    len(path_world)
                    * progress
                )
            )

            count = min(
                count,
                len(path_world)
            )

        else:

            count = len(
                path_world
            )

        path_x = [
            point[0]
            for point in path_world[:count]
        ]

        path_y = [
            point[1]
            for point in path_world[:count]
        ]

        path_line.set_data(
            path_x,
            path_y
        )

        return [
            path_line
        ]

    # --------------------------------------------------------
    # Create animation
    # --------------------------------------------------------

    animation = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=1000 / FPS,
        blit=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    filename = os.path.join(
        FIGURES_DIR,
        "astar_animation.gif"
    )

    save_gif(
        animation,
        filename
    )

    plt.close(
        fig
    )

    print(
        f"A* nodes explored: {nodes_explored}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "MOTION PLANNER ANIMATIONS"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Generate animations
    # --------------------------------------------------------

    animate_rrt()

    animate_rrt_star()

    animate_prm()

    animate_astar()

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "ALL ANIMATIONS COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        "\nAnimations saved to:"
    )

    print(
        f"  {FIGURES_DIR}"
    )