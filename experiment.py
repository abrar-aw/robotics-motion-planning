import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import os
import statistics


# ============================================================
# IMPORT PLANNERS
# ============================================================

import rrt
import rrt_star
import prm
import astar


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

NUM_TRIALS = 20

PRM_SAMPLES = 500
PRM_CONNECTION_RADIUS = 2.0

ASTAR_GRID_SIZE = 100

RRT_STAR_ITERATIONS = 5000

RESULTS_FILE = "experiment_results.csv"
SUMMARY_FILE = "experiment_summary.csv"

RRT_STAR_CONVERGENCE_FILE = "rrt_star_convergence.csv"
PRM_SAMPLING_FILE = "prm_sampling_experiment.csv"

FIGURE_DIR = "figures"

os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def mean(values):
    """Return the arithmetic mean."""
    if not values:
        return 0.0

    return statistics.mean(values)


def std(values):
    """Return sample standard deviation."""
    if len(values) <= 1:
        return 0.0

    return statistics.stdev(values)


def median(values):
    """Return the median."""
    if not values:
        return 0.0

    return statistics.median(values)


def save_csv(filename, fieldnames, rows):
    """Save a list of dictionaries to CSV."""

    with open(
        filename,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# RRT
# ============================================================

def run_rrt(seed):

    np.random.seed(seed)

    start_time = time.perf_counter()

    tree, goal_node, iterations = rrt.rrt()

    computation_time = (
        time.perf_counter() - start_time
    )

    if goal_node is not None:

        path = rrt.get_path(goal_node)

        length = rrt.path_length(path)

        return {
            "success": True,
            "path_length": length,
            "time": computation_time,
            "nodes": len(tree),
            "iterations": iterations
        }

    return {
        "success": False,
        "path_length": None,
        "time": computation_time,
        "nodes": len(tree),
        "iterations": iterations
    }


# ============================================================
# RRT*
# ============================================================

def run_rrt_star(seed):

    np.random.seed(seed)

    start_time = time.perf_counter()

    tree, goal_node, iterations = (
        rrt_star.rrt_star()
    )

    computation_time = (
        time.perf_counter() - start_time
    )

    if goal_node is not None:

        path = rrt_star.get_path(goal_node)

        length = rrt_star.path_length(path)

        return {
            "success": True,
            "path_length": length,
            "time": computation_time,
            "nodes": len(tree),
            "iterations": iterations
        }

    return {
        "success": False,
        "path_length": None,
        "time": computation_time,
        "nodes": len(tree),
        "iterations": iterations
    }


# ============================================================
# PRM
# ============================================================

def run_prm(seed):

    np.random.seed(seed)

    start_time = time.perf_counter()

    (
        samples,
        graph,
        path,
        path_cost
    ) = prm.prm(
        num_samples=PRM_SAMPLES,
        connection_radius=PRM_CONNECTION_RADIUS
    )

    computation_time = (
        time.perf_counter() - start_time
    )

    if path is not None:

        length = prm.path_length(path)

        return {
            "success": True,
            "path_length": length,
            "time": computation_time,
            "nodes": len(graph),
            "iterations": PRM_SAMPLES
        }

    return {
        "success": False,
        "path_length": None,
        "time": computation_time,
        "nodes": len(graph),
        "iterations": PRM_SAMPLES
    }


# ============================================================
# A*
# ============================================================

def run_astar(seed):

    # A* is deterministic for a fixed grid.
    np.random.seed(seed)

    start_time = time.perf_counter()

    # Include grid construction in the timing.
    # This makes the measurement more comparable with
    # PRM/RRT/RRT*, which include their environment setup
    # and planning operations.

    grid = astar.create_grid()

    start_grid = astar.world_to_grid(
        astar.START
    )

    goal_grid = astar.world_to_grid(
        astar.GOAL
    )

    path, grid_cost, nodes_explored = (
        astar.astar(
            grid,
            start_grid,
            goal_grid
        )
    )

    computation_time = (
        time.perf_counter() - start_time
    )

    if path is not None:

        length = astar.path_length(path)

        return {
            "success": True,
            "path_length": length,
            "time": computation_time,
            "nodes": nodes_explored,
            "iterations": nodes_explored
        }

    return {
        "success": False,
        "path_length": None,
        "time": computation_time,
        "nodes": nodes_explored,
        "iterations": nodes_explored
    }


# ============================================================
# MAIN COMPARISON EXPERIMENT
# ============================================================

def run_comparison_experiment():

    algorithms = {
        "RRT": run_rrt,
        "RRT*": run_rrt_star,
        "PRM": run_prm,
        "A*": run_astar
    }

    raw_results = []

    print()
    print("=" * 75)
    print("SAMPLING-BASED AND GRID-BASED MOTION PLANNING")
    print("COMPARATIVE EXPERIMENT")
    print("=" * 75)

    print(f"\nNumber of trials: {NUM_TRIALS}")
    print()

    # --------------------------------------------------------
    # Run all algorithms
    # --------------------------------------------------------

    for algorithm_name, algorithm_function in algorithms.items():

        print()
        print(f"Running {algorithm_name}...")
        print("-" * 50)

        for trial in range(
            1,
            NUM_TRIALS + 1
        ):

            result = algorithm_function(trial)

            raw_results.append(
                {
                    "algorithm": algorithm_name,
                    "trial": trial,
                    "success": int(result["success"]),
                    "path_length": (
                        ""
                        if result["path_length"] is None
                        else f"{result['path_length']:.6f}"
                    ),
                    "computation_time": (
                        f"{result['time']:.6f}"
                    ),
                    "nodes": result["nodes"],
                    "iterations": result["iterations"]
                }
            )

            if result["success"]:

                print(
                    f"Trial {trial:2d}: "
                    f"path = {result['path_length']:.3f} | "
                    f"time = {result['time']:.4f}s | "
                    f"nodes = {result['nodes']}"
                )

            else:

                print(
                    f"Trial {trial:2d}: "
                    f"NO PATH | "
                    f"time = {result['time']:.4f}s"
                )

    # --------------------------------------------------------
    # Save raw data
    # --------------------------------------------------------

    save_csv(
        RESULTS_FILE,
        [
            "algorithm",
            "trial",
            "success",
            "path_length",
            "computation_time",
            "nodes",
            "iterations"
        ],
        raw_results
    )

    print()
    print(
        f"Raw experimental data saved to: "
        f"{RESULTS_FILE}"
    )

    # --------------------------------------------------------
    # Generate summary
    # --------------------------------------------------------

    summary_rows = []

    algorithms_list = [
        "RRT",
        "RRT*",
        "PRM",
        "A*"
    ]

    for algorithm_name in algorithms_list:

        algorithm_results = [
            row
            for row in raw_results
            if row["algorithm"] == algorithm_name
        ]

        successful_results = [
            row
            for row in algorithm_results
            if row["success"] == 1
        ]

        path_lengths = [
            float(row["path_length"])
            for row in successful_results
        ]

        times = [
            float(row["computation_time"])
            for row in algorithm_results
        ]

        nodes = [
            int(row["nodes"])
            for row in algorithm_results
        ]

        success_rate = (
            len(successful_results)
            / NUM_TRIALS
            * 100
        )

        summary_rows.append(
            {
                "algorithm": algorithm_name,

                "success_rate_percent":
                    f"{success_rate:.2f}",

                "mean_path_length":
                    f"{mean(path_lengths):.4f}",

                "std_path_length":
                    f"{std(path_lengths):.4f}",

                "median_path_length":
                    f"{median(path_lengths):.4f}",

                "mean_computation_time":
                    f"{mean(times):.6f}",

                "std_computation_time":
                    f"{std(times):.6f}",

                "median_computation_time":
                    f"{median(times):.6f}",

                "mean_nodes":
                    f"{mean(nodes):.2f}",

                "std_nodes":
                    f"{std(nodes):.2f}"
            }
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    save_csv(
        SUMMARY_FILE,
        [
            "algorithm",
            "success_rate_percent",
            "mean_path_length",
            "std_path_length",
            "median_path_length",
            "mean_computation_time",
            "std_computation_time",
            "median_computation_time",
            "mean_nodes",
            "std_nodes"
        ],
        summary_rows
    )

    # --------------------------------------------------------
    # Print final summary
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("FINAL EXPERIMENT SUMMARY")
    print("=" * 90)

    print(
        f"{'Algorithm':<12}"
        f"{'Success':>10}"
        f"{'Mean Path':>14}"
        f"{'Std Path':>13}"
        f"{'Mean Time':>15}"
        f"{'Median Time':>15}"
        f"{'Mean Nodes':>14}"
    )

    print("-" * 90)

    for row in summary_rows:

        print(
            f"{row['algorithm']:<12}"
            f"{row['success_rate_percent']:>9}%"
            f"{row['mean_path_length']:>14}"
            f"{row['std_path_length']:>13}"
            f"{row['mean_computation_time']:>15}"
            f"{row['median_computation_time']:>15}"
            f"{row['mean_nodes']:>14}"
        )

    print("=" * 90)

    # --------------------------------------------------------
    # Identify unusual runtime values
    # --------------------------------------------------------

    rrt_star_times = [
        float(row["computation_time"])
        for row in raw_results
        if row["algorithm"] == "RRT*"
    ]

    rrt_star_median = median(rrt_star_times)

    outliers = [
        (i + 1, t)
        for i, t in enumerate(rrt_star_times)
        if t > 2 * rrt_star_median
    ]

    if outliers:

        print()
        print("RRT* runtime observations:")
        print(
            f"Median runtime: "
            f"{rrt_star_median:.4f} s"
        )

        for trial, runtime in outliers:

            print(
                f"  Trial {trial}: "
                f"{runtime:.4f} s "
                f"(unusually high)"
            )

    print()

    return raw_results, summary_rows


# ============================================================
# COMPARISON FIGURES
# ============================================================

def plot_comparison(
    raw_results,
    summary_rows
):

    algorithms = [
        "RRT",
        "RRT*",
        "PRM",
        "A*"
    ]

    x = np.arange(
        len(algorithms)
    )

    # --------------------------------------------------------
    # Extract summary statistics
    # --------------------------------------------------------

    mean_lengths = []
    std_lengths = []

    mean_times = []
    std_times = []

    median_times = []

    success_rates = []

    for algorithm in algorithms:

        row = next(
            row
            for row in summary_rows
            if row["algorithm"] == algorithm
        )

        mean_lengths.append(
            float(row["mean_path_length"])
        )

        std_lengths.append(
            float(row["std_path_length"])
        )

        mean_times.append(
            float(row["mean_computation_time"])
        )

        std_times.append(
            float(row["std_computation_time"])
        )

        median_times.append(
            float(row["median_computation_time"])
        )

        success_rates.append(
            float(row["success_rate_percent"])
        )

    # ========================================================
    # 1. PATH LENGTH
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        x,
        mean_lengths,
        yerr=std_lengths,
        capsize=6
    )

    plt.xticks(
        x,
        algorithms,
        fontsize=11
    )

    plt.ylabel(
        "Path Length",
        fontsize=12
    )

    plt.xlabel(
        "Algorithm",
        fontsize=12
    )

    plt.title(
        "Average Path Length Comparison",
        fontsize=14
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "01_path_length_comparison.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # ========================================================
    # 2. COMPUTATION TIME
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        x,
        mean_times,
        yerr=std_times,
        capsize=6
    )

    # Logarithmic scale makes the large difference between
    # algorithms much easier to see.
    plt.yscale("log")

    plt.xticks(
        x,
        algorithms,
        fontsize=11
    )

    plt.ylabel(
        "Computation Time (s, log scale)",
        fontsize=12
    )

    plt.xlabel(
        "Algorithm",
        fontsize=12
    )

    plt.title(
        "Average Computation Time Comparison",
        fontsize=14
    )

    plt.grid(
        axis="y",
        alpha=0.25,
        which="both"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "02_computation_time_comparison.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # ========================================================
    # 3. SUCCESS RATE
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        x,
        success_rates
    )

    plt.xticks(
        x,
        algorithms,
        fontsize=11
    )

    plt.ylabel(
        "Success Rate (%)",
        fontsize=12
    )

    plt.xlabel(
        "Algorithm",
        fontsize=12
    )

    plt.ylim(
        0,
        105
    )

    plt.title(
        "Motion Planning Success Rate",
        fontsize=14
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "03_success_rate_comparison.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # ========================================================
    # 4. PATH LENGTH DISTRIBUTION
    # ========================================================

    path_data = []

    for algorithm in algorithms:

        values = [
            float(row["path_length"])
            for row in raw_results
            if (
                row["algorithm"] == algorithm
                and row["success"] == 1
            )
        ]

        path_data.append(values)

    plt.figure(
        figsize=(9, 6)
    )

    plt.boxplot(
        path_data,
        tick_labels=algorithms
    )

    plt.ylabel(
        "Path Length",
        fontsize=12
    )

    plt.xlabel(
        "Algorithm",
        fontsize=12
    )

    plt.title(
        "Path Length Distribution Across Trials",
        fontsize=14
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "04_path_length_distribution.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # ========================================================
    # 5. RUNTIME DISTRIBUTION
    # ========================================================

    time_data = []

    for algorithm in algorithms:

        values = [
            float(row["computation_time"])
            for row in raw_results
            if row["algorithm"] == algorithm
        ]

        time_data.append(values)

    plt.figure(
        figsize=(9, 6)
    )

    plt.boxplot(
        time_data,
        tick_labels=algorithms
    )

    plt.yscale("log")

    plt.ylabel(
        "Computation Time (s, log scale)",
        fontsize=12
    )

    plt.xlabel(
        "Algorithm",
        fontsize=12
    )

    plt.title(
        "Computation Time Distribution Across Trials",
        fontsize=14
    )

    plt.grid(
        axis="y",
        alpha=0.25,
        which="both"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "05_computation_time_distribution.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# RRT* CONVERGENCE EXPERIMENT
# ============================================================

def run_rrt_star_convergence():

    iteration_values = [
        500,
        1000,
        2000,
        3000,
        4000,
        5000
    ]

    path_lengths = []
    computation_times = []

    print()
    print("=" * 65)
    print("RRT* CONVERGENCE EXPERIMENT")
    print("=" * 65)

    for iterations in iteration_values:

        # Same seed ensures that the experiment starts
        # from the same random sequence.
        np.random.seed(42)

        start_time = time.perf_counter()

        tree, goal_node, _ = (
            rrt_star.rrt_star(
                max_iterations=iterations
            )
        )

        computation_time = (
            time.perf_counter() - start_time
        )

        if goal_node is not None:

            path = rrt_star.get_path(
                goal_node
            )

            length = rrt_star.path_length(
                path
            )

            path_lengths.append(length)

            print(
                f"{iterations:5d} iterations | "
                f"path = {length:.3f} | "
                f"time = {computation_time:.4f}s"
            )

        else:

            path_lengths.append(np.nan)

            print(
                f"{iterations:5d} iterations | "
                f"NO PATH | "
                f"time = {computation_time:.4f}s"
            )

        computation_times.append(
            computation_time
        )

    # --------------------------------------------------------
    # Save data
    # --------------------------------------------------------

    rows = []

    for i in range(
        len(iteration_values)
    ):

        rows.append(
            {
                "iterations":
                    iteration_values[i],

                "path_length":
                    path_lengths[i],

                "computation_time":
                    computation_times[i]
            }
        )

    save_csv(
        RRT_STAR_CONVERGENCE_FILE,
        [
            "iterations",
            "path_length",
            "computation_time"
        ],
        rows
    )

    # --------------------------------------------------------
    # Plot convergence
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        iteration_values,
        path_lengths,
        marker="o",
        linewidth=2
    )

    plt.xlabel(
        "Number of Iterations",
        fontsize=12
    )

    plt.ylabel(
        "Path Length",
        fontsize=12
    )

    plt.title(
        "RRT* Path Length Convergence",
        fontsize=14
    )

    plt.grid(
        True,
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "06_rrt_star_convergence.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# PRM SAMPLING EXPERIMENT
# ============================================================

def run_prm_sampling_experiment():

    sample_values = [
        100,
        200,
        300,
        500,
        750,
        1000
    ]

    path_lengths = []
    computation_times = []

    print()
    print("=" * 65)
    print("PRM SAMPLING DENSITY EXPERIMENT")
    print("=" * 65)

    for samples in sample_values:

        np.random.seed(42)

        start_time = time.perf_counter()

        (
            sample_points,
            graph,
            path,
            path_cost
        ) = prm.prm(
            num_samples=samples,
            connection_radius=PRM_CONNECTION_RADIUS
        )

        computation_time = (
            time.perf_counter() - start_time
        )

        if path is not None:

            length = prm.path_length(path)

            path_lengths.append(length)

            print(
                f"{samples:5d} samples | "
                f"path = {length:.3f} | "
                f"time = {computation_time:.4f}s"
            )

        else:

            path_lengths.append(np.nan)

            print(
                f"{samples:5d} samples | "
                f"NO PATH | "
                f"time = {computation_time:.4f}s"
            )

        computation_times.append(
            computation_time
        )

    # --------------------------------------------------------
    # Save data
    # --------------------------------------------------------

    rows = []

    for i in range(
        len(sample_values)
    ):

        rows.append(
            {
                "samples":
                    sample_values[i],

                "path_length":
                    path_lengths[i],

                "computation_time":
                    computation_times[i]
            }
        )

    save_csv(
        PRM_SAMPLING_FILE,
        [
            "samples",
            "path_length",
            "computation_time"
        ],
        rows
    )

    # --------------------------------------------------------
    # Plot PRM path quality
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        sample_values,
        path_lengths,
        marker="o",
        linewidth=2
    )

    plt.xlabel(
        "Number of Samples",
        fontsize=12
    )

    plt.ylabel(
        "Path Length",
        fontsize=12
    )

    plt.title(
        "PRM Path Length vs Sampling Density",
        fontsize=14
    )

    plt.grid(
        True,
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "07_prm_sampling.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # --------------------------------------------------------
    # Plot PRM computation time
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        sample_values,
        computation_times,
        marker="o",
        linewidth=2
    )

    plt.xlabel(
        "Number of Samples",
        fontsize=12
    )

    plt.ylabel(
        "Computation Time (s)",
        fontsize=12
    )

    plt.title(
        "PRM Computation Time vs Sampling Density",
        fontsize=14
    )

    plt.grid(
        True,
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "08_prm_sampling_time.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. Main comparison
    # --------------------------------------------------------

    raw_results, summary_rows = (
        run_comparison_experiment()
    )

    # --------------------------------------------------------
    # 2. Generate comparison figures
    # --------------------------------------------------------

    plot_comparison(
        raw_results,
        summary_rows
    )

    # --------------------------------------------------------
    # 3. RRT* convergence
    # --------------------------------------------------------

    run_rrt_star_convergence()

    # --------------------------------------------------------
    # 4. PRM sampling density
    # --------------------------------------------------------

    run_prm_sampling_experiment()

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 65)

    print()
    print("CSV files:")
    print(f"  {RESULTS_FILE}")
    print(f"  {SUMMARY_FILE}")
    print(f"  {RRT_STAR_CONVERGENCE_FILE}")
    print(f"  {PRM_SAMPLING_FILE}")

    print()
    print("Figures:")
    print(f"  {FIGURE_DIR}/")

    print()