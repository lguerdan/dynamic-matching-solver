## Setup

To run the code, install numpy, pandas, matplotlib, seaborn, and gurobi. You will need to [download the gurobi optimizer](https://www.gurobi.com/downloads/) locally, then configure an academic license. 


## Code structure

You will see several common variables in the code related to the problem formulation. These are: 

- `I`: the number of resources (i=0 is for no match)
- `J`: the number of arriving agents
- `d`: the number of rounds each agent will leave before exit
- `T`: the total time rounds in the simulation (T=J+d)
- `c`: an array specifying the number of each copies for each resource
- `k`: an array specifying the wait time of each resource
- `valid_matches`: an (I,J,T) array specifying which agents will be present at each time given arrivals and the agent wait period. 
- `pairing_weights`: an (I,J,T) array specifying the utility of matching an agent and resource at a given time

Included files:
- Linear program solvers and utility functions are defined in [dynamic_matching_solvers.py](dynamic_matching_solvers.py): there is a primal solver, dual solver, greedy solver, and online-dual solver. 
- [dynamic_matching_experiments.ipynb](dynamic_matching_experiments.ipynb) runs experiments based on the solvers implemented in the main file. 
- [dynamic_matching_solver_sim_forward.ipynb](dynamic_matching_solver_sim_forward.ipynb) contains an experimental implementation of simulating future arrivals, and making allocations based on these results. This implementation has not been extensively tested, and needs to be modified such that:
    1. The primal solver runs a comparison for each i,t in the simulation interval to decide whether an allocation should occur.
    2. A dual solver creates an average alpha vector from many runs, and allocates based on this vector. 

To create a scenario and run the solvers, run: 

```
J=6
I=3
d=2
T = J+d 

k = np.array([1,3,3])
c = np.array([J,1, 1])

valid_matches, pairing_weights = create_simulation_scenario(I,J,T)


objp, allocs = primal_solutions(valid_matches, pairing_weights, I, J, T, k, c)
objd, alphas, betas = dual_solutions(valid_matches,pairing_weights, I, J, T, k, c)
objo, online_allocations = online_matching(I,J,T,k,c,alphas,betas,allocs,valid_matches,pairing_weights)
objg, greedy_allocs = greedy_matching(I,J,T,k,c,d,valid_matches,pairing_weights)
```

You can then visualize allocations using `display_3D`.



