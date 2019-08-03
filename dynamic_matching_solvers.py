import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from gurobi import *


###########
## Primal-Dual solver utility functions
###########
def in_constraint(v):
    if v[1]:
        return True
    else:
        return False

def make_alpha_mapping(I,J,T,alphas,valid_matches):
    '''Creates an index into the alpha array for each position in valid matches'''
    constraints_d, _ = dual_constraint_matrix(valid_matches,pairing_weights,I,J,T,k)
    constraints_d = constraints_d[:,:alphas.size]
    alpha_map = np.zeros((*valid_matches.shape,constraints_d.shape[1]),dtype=np.bool)

    cix=0
    for i in range(I):
            for j in range(J):
                for t in range(T):
                    if valid_matches[i][j][t]:
                        alpha_map[i,j,t,:] = constraints_d[cix,:]
                        cix += 1
    return alpha_map

def sum_alpha_it(I,T, alphas,i,t,k):
    '''Return the alpha terms summed at a given i,t'''
    alpha_it = alphas.reshape(I,T)
    startit = max(t-k[i]+1,0)
    return np.sum(alpha_it[i,startit:t+1])


###########
## Primal-Dual solver functions
###########

########
## To make the primal constraint matrix: 
##  Apply constraint equations for each position in valid matches, then flatten to a 2D form understood by Gurobi 
#######
def primal_constraint_matrix(valid_matches,I,J,T,k):

    constraints = np.zeros((T*I+J,valid_matches.size),dtype=np.float128)
    cix = 0
    
    #constraints limiting to one resource allocation in the time interval
    for i in range(I):
        for t in range(T):
            constraint = np.zeros((I,J,T), np.int)
            valid_mask = constraint.copy()
            endix = min(t+k[i],T)
            valid_mask[i,:,t:endix] = 1 
            constraint[(valid_mask == 1)] = 1
            constraints[cix,:] = constraint.reshape((1, constraint.shape[0] * constraint.shape[1] * constraint.shape[2]))
            cix += 1

    #constraints limiting each agent to only match once            
    for j in range(J):
        constraint = np.zeros((I,J,T), np.int)
        valid_mask = constraint.copy()
        valid_mask[1:,j,:] = 1

        constraint[(valid_matches == 1) & (valid_mask ==1)] = 1
        constraints[cix+j,:] = constraint.reshape((1, constraint.shape[0] * constraint.shape[1] * constraint.shape[2]))
    
    return constraints

########
## To make the dual constraint matrix: 
##  Create a constraint map to see which alphas/betas apply at a given location in the primal. 
##   Each valid location will correspond with a constraint in the dual, and the variables==1 will be those in the `cmap[i][j][t]`
#######
def dual_constraint_matrix(valid_matches,pairing_weights,I,J,T,k):
    '''
    Dual constraint matrix: Number IJT positions * number dual variables
             - Each row corresponds to an IJT position in the grid
             - Each column corresponds to a dual variable 
    '''
    num_positions = I*J*T
    num_primal_constraints = I*T+J
    dual_constraint_matrix = np.zeros((num_positions, num_primal_constraints))
    
    inequalities = np.zeros(num_positions)
    constraint_map = np.zeros((I,J,T,num_primal_constraints), np.int) 
    
    cix = 0

    #constraints limiting to one resource allocation in the time interval
    for i in range(I):
        for t in range(T):
            constraint = np.zeros((I,J,T), np.int)
            valid_mask = constraint.copy()

            endix = min(t+k[i],T)
            valid_mask[i,:,t:endix] = 1 
            constraint[(valid_mask == 1)] = 1
            constraint_map[:,:,:,cix] = constraint.copy()
            cix += 1

    #constraints limiting each agent to only match once            
    for j in range(J):
        constraint = np.zeros((I,J,T), np.int)
        valid_mask = constraint.copy()
        valid_mask[1:,j,:] = 1
        constraint[valid_mask ==1] = 1
        constraint_map[:,:,:,cix] = constraint.copy()
        cix += 1
    
    cix = 0
    for i in range(I):
        for j in range(J):
            for t in range(T):
                dual_constraint_matrix[cix,:] = constraint_map[i,j,t,:] 
                inequalities[cix] = pairing_weights[i,j,t]
                cix += 1
    
    return dual_constraint_matrix, inequalities

def primal_solutions(valid_matches, pairing_weights, I, J, T, k, c):
    '''
    Sets up and solves the primal linear program using the gurobi library API. 
    Defers most business logic for for primal_constraint_matrix
    '''
        
    m = Model("dynamicmatch_primal")
    m.modelSense = GRB.MAXIMIZE
    m.setParam( 'OutputFlag', False )
    m.setParam( 'NumericFocus', 3)
    
    weights = pairing_weights.reshape(pairing_weights.shape[0] * pairing_weights.shape[1] * pairing_weights.shape[2])
    constraints = primal_constraint_matrix(valid_matches,I,J,T,k)
    
    keys = range(constraints.shape[1])
    variables = m.addVars(keys,
                    vtype=GRB.CONTINUOUS,
                     obj=weights,
                     name="primal",
                     lb=0)
    
    #Add inequality condition based on number of copies
    for cix, constraint in enumerate(constraints):
        equality = c[cix // T] if cix < T * I else 1
        m.addConstr(sum(variables[o]*c for o,c in filter(in_constraint, zip(variables,constraint))) <= equality)

    m.optimize()
    allocations = np.array([variables[var].X for var in variables], dtype=np.float128).reshape(pairing_weights.shape)

    return m.objVal, allocations


def dual_solutions(valid_matches, pairing_weights, I, J, T, k, c):
    '''
    Sets up and solves the dual linear program using the gurobi library API. 
    Defers most business logic for for dual_constraint_matrix
    '''
    md = Model("dynamicmatch_dual")
    md.modelSense = GRB.MINIMIZE
    md.setParam( 'OutputFlag', False )
    md.setParam( 'NumericFocus', 3)

    constraints_d, inequalities = dual_constraint_matrix(valid_matches,pairing_weights,I,J,T,k)
    variables_d = np.ones(constraints_d.shape[1])
    
    for ix in range(constraints_d.shape[1]):
        variables_d[ix] = c[ix // T] if ix < T * I else 1
    
    keys = range(constraints_d.shape[1])
    variables = md.addVars(keys,
                    vtype=GRB.CONTINUOUS,
                    obj=variables_d,
                    name="dual",
                    lb=0)
    
    #Add dual coefficient based on number of copies
    for cix, constraint in enumerate(constraints_d):
        constr = inequalities[cix] + sum( variables[o]*-1*c for o,c in filter(in_constraint, zip(variables,constraint))) <= 0
        md.addConstr(constr)
        
    md.optimize()
    duals = np.array([variables[var].X for var in variables],dtype=np.longdouble)
    betas = duals[duals.size - J:]
    alphas = duals[:duals.size - J]
    
    return md.objVal, alphas, betas

###########
## Online-Dual solver functions
###########
def online_matching(I,J,T,k,c,alphas,betas,allocs,valid_matches,pairing_weights):
    ''' Perform online matching algorithm using alpha and beta dual variables'''

    online_allocations = np.zeros(pairing_weights.shape)
    utility = 0
    candidate_matches = valid_matches.copy()
    
    resource_uses = np.zeros((pairing_weights.shape[0],pairing_weights.shape[2]))

    for t in range(T):
        for j in range(J):
    
            if candidate_matches[0,j,t]:

                alpha_sums = np.array([sum_alpha_it(I,T,alphas,i,t,k) for i in range(I)])
                betai = betas[j] * np.ones(alpha_sums.shape)       
                comps = pairing_weights[:,j,t] - alpha_sums - betai

                alloc_ix = get_allocation_from_comparison(comps, pairing_weights[:,j,t], c, resource_uses[:,t] )
                
                #if resource already in use wait for it to become free
                if alloc_ix != -1:
                    
                    online_allocations[alloc_ix, j, t] = 1
                    utility += pairing_weights[alloc_ix,j,t]
                    
                    #Agent only leaves the market if it doesn't self match
                    if alloc_ix != 0:
                        candidate_matches[:,j,:] = 0
                    
                    resource_uses[alloc_ix,t:t+k[alloc_ix]] += 1

    return utility, online_allocations


def get_allocation_from_comparison(comps, weights, c,resource_uses ):
    '''Returns max comparison index, max weight if tie'''
    w = weights.copy()
    maxval = np.max(comps)
    
    #tie case
    if sum(comps == maxval) > 1:
        tie_ixs = np.zeros(comps.shape, np.bool)
        tie_ixs[comps == maxval] = 1
        w[~tie_ixs] = -100000
        alloc_ix = np.argmax(w)
    else: 
        alloc_ix = np.argmax(comps)
    
    if resource_uses[alloc_ix] >= c[alloc_ix]:
        return -1
    
    return alloc_ix


###########
## Greedy solver functions
###########
def greedy_matching(I,J,T,k,c,d,valid_matches,pairing_weights):
    '''Greedily match arriving agents based on the max pairing weight'''
    greedy_allocations = np.zeros(pairing_weights.shape)
    utility = 0
    candidate_matches = valid_matches.copy()
    resource_uses = np.zeros((pairing_weights.shape[0],pairing_weights.shape[2]))

    for t in range(T):
        
        candidate_matches_t = np.array(candidate_matches[:,:,t])
        
        while(True):  
            matches_t, weights_t = get_possible_matches(I,J,T,c,resource_uses[:,t], candidate_matches_t, pairing_weights[:,:,t])
            
            # no more matches can be made at this time
            if matches_t.sum() == 0:
                break

            i_alloc,j_alloc = np.unravel_index(weights_t.argmax(), weights_t.shape)
            
            # Only self alloc if it is the last time before departure
            if t < j_alloc + d and i_alloc == 0:
                candidate_matches_t[:,j_alloc] = 0

            else: 
                greedy_allocations[i_alloc,j_alloc,t] = 1
                utility += pairing_weights[i_alloc,j_alloc,t]

                resource_uses[i_alloc,t:t+k[i_alloc]] += 1
                candidate_matches[:,j_alloc,:] = 0
                candidate_matches_t[:,j_alloc] = 0
            
    return utility, greedy_allocations  
                
def get_possible_matches(I,J,T,c, resource_uses, candidate_matches, pairing_weights):
    '''Return possible matches given arrival model and available resources'''
    
    candidate_matches_t = candidate_matches.copy()
    resource_uses_t = np.repeat((resource_uses < c).reshape(I,1),J, axis=1)
    
    candidate_matches_t[~(resource_uses_t)] = 0 
    pairing_weights_t =  np.array(pairing_weights)
    pairing_weights_t[0,:] = 1e-5
    pairing_weights_t[candidate_matches_t == 0] = 0
      
    return candidate_matches_t, pairing_weights_t
