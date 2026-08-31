Sampling-Based Motion Planning for Mobile Robots

This project investigates and compares motion planning algorithms for mobile robots navigating environments with obstacles. The project focuses on three sampling-based planners: Rapidly-exploring Random Tree (RRT), RRT*, and Probabilistic Roadmap (PRM), and includes A* as a grid-based baseline for comparison.

The algorithms are implemented in Python and evaluated experimentally using a common obstacle environment. Performance is compared using path length, computation time, success rate, and number of explored/generated nodes.

Additional experiments investigate the convergence behavior of RRT* as the number of iterations increases and the effect of sampling density on PRM performance.

Algorithms  
RRT, Rapidly-exploring Random Tree  
RRT*, Optimal variant of RRT with rewiring  
PRM, Probabilistic Roadmap  
A*, Grid-based heuristic search algorithm  
Evaluation

The planners are evaluated based on:

Path length  
Computation time  
Success rate  
Number of nodes explored/generated  
RRT* convergence with increasing iterations  
PRM performance with varying sampling density

The project also includes experimental CSV data and generated visualizations for analysis.
