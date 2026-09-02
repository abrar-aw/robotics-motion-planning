"""Shared environment configuration for all motion planners."""

# Workspace limits
X_LIMITS = (0, 10)
Y_LIMITS = (0, 10)

# Start and goal positions
START = (1, 1)
GOAL = (9, 9)

# Obstacles are represented as:
# (x, y, width, height)
#
# This map uses varied obstacle sizes and positions to create
# a more interesting planning problem while keeping clear
# passages between obstacles.
OBSTACLES = [
    (2.5, 1.8, 1.5, 2.8),
    (5.0, 1.0, 1.3, 2.5),
    (3.2, 5.0, 2.2, 1.5),
    (6.8, 5.8, 1.4, 2.8),
    (4.8, 7.8, 1.8, 1.2),
]
