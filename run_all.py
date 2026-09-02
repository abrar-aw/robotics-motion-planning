import subprocess
import sys


FILES = [
    "rrt.py",
    "rrt_star.py",
    "prm.py",
    "astar.py",
    "experiment.py"
]


def main():
    for filename in FILES:
        print("\n" + "=" * 60)
        print(f"Running {filename}")
        print("=" * 60)

        result = subprocess.run(
            [sys.executable, filename]
        )

        if result.returncode != 0:
            print(f"\nERROR: {filename} failed.")
            print("Stopping execution.")
            sys.exit(result.returncode)

        print(f"\n{filename} completed successfully.")

    print("\n" + "=" * 60)
    print("ALL PROGRAMS COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()