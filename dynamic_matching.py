import numpy as np

from scipy.optimize import linprog
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from gurobi import *

class DynamicMatchingBase:
    def __init__(self,I,J,T,d,k,c):
        self.I = I
        self.J = J
        self.T = T
        self.use_times = k
        self.num_copies = c
        self.wait_time = d
        
        self.create_tableau()       

    ##Instance utility functions
    def create_tableau(self):
        valid_matches = np.zeros((self.I,self.J,self.T), np.int)

        #valid allocations based on arival time
        for t in range(self.T):
            valid_matches[:,max(0,t-self.wait_time):min(t+1,self.J),t] = 1

        #make weights uniform across time
        pairing_weights = np.random.random(valid_matches.shape) 
        for t in range(1,pairing_weights.shape[2]):
            pairing_weights[:,:,t] = pairing_weights[:,:,0]
    
        pairing_weights[0,:,:] = .000001
        pairing_weights[valid_matches == 0] = -1
        
        self.valid_matches = valid_matches
        self.pairing_weights = pairing_weights
    
    def make_alpha_mapping(self,alphas):
        constraints_d, _ = self.dual_constraint_matrix()
        constraints_d = constraints_d[:,:alphas.size]
        alpha_map = np.zeros((*self.valid_matches.shape,constraints_d.shape[1]))

        cix=0
        for i in range(self.I):
                for j in range(self.J):
                    for t in range(self.T):
                        if self.valid_matches[i][j][t]:
                            alpha_map[i,j,t,:] = constraints_d[cix,:]
                            cix += 1
        return alpha_map  
    
    @staticmethod
    def in_constraint(v):
        if v[1]:
            return True
        else:
            return False
    
    @staticmethod
    def print_tableau(matches):
        for i in range(matches.shape[2]):
            print(matches[:,:,i])
    
    def validate_allocation(self, allocations):
        agents_matched = allocations.sum(axis=2).sum(axis=0).sum()
        print("{} out of {} agents matched".format(agents_matched, J))
        print("{} agents have self matched".format(allocations[0,:,:].sum()))

        assert agents_matched <= J, "An agent has been matched more than once"

        for i in range(I):
            for j in range(J):
                for t in range(T):
                    maxit = min(T, t+d)
                    assert allocations[i,j,t:maxit].sum() < c[i], "A resource has been over allocated"

    def primal_constraint_matrix(self):

        constraints = np.zeros((self.valid_matches.sum()+self.J,self.valid_matches.size))
        cix = 0
        #constraints limiting to one resource allocation in the time interval
        for i in range(self.I):
            for t in range(self.T):
                constraint = np.zeros((self.I,self.J,self.T), np.int)
                valid_mask = constraint.copy()
                endix = min(t+self.use_times[i],self.T)
                valid_mask[i,:,t:endix] = 1 
                constraint[(self.valid_matches == 1) & (valid_mask == 1)] = 1
                constraints[cix,:] = constraint.reshape((1, constraint.shape[0] * constraint.shape[1] * constraint.shape[2]))
                cix += 1

        #constraints limiting each agent to only match once            
        for j in range(self.J):
            constraint = np.zeros((self.I,self.J,self.T), np.int)
            valid_mask = constraint.copy()
            valid_mask[:,j,:] = 1

            constraint[(self.valid_matches == 1) & (valid_mask ==1)] = 1
            constraints[cix+j,:] = constraint.reshape((1, constraint.shape[0] * constraint.shape[1] * constraint.shape[2]))

        return constraints

    
    def dual_constraint_matrix(self):

        constraint_map = np.zeros((self.I,self.J,self.T,self.T*self.I+self.J), np.int)
        cix = 0

        #constraints limiting to one resource allocation in the time interval
        for i in range(self.I):
            for t in range(self.T):
                constraint = np.zeros((self.I,self.J,self.T), np.int)
                valid_mask = constraint.copy()

                endix = min(t+self.use_times[i],self.T)
                valid_mask[i,:,t:endix] = 1 
                constraint[(self.valid_matches == 1) & (valid_mask == 1)] = 1

                constraint_map[:,:,:,cix] = constraint.copy()
                cix += 1

        #constraints limiting each agent to only match once            
        for j in range(self.J):
            constraint = np.zeros((self.I,self.J,self.T), np.int)
            valid_mask = constraint.copy()
            valid_mask[:,j,:] = 1
            constraint[(self.valid_matches == 1) & (valid_mask ==1)] = 1
            constraint_map[:,:,:,cix] = constraint.copy()
            cix += 1

        constraint_matrix = np.zeros((self.valid_matches.sum(), constraint_map.shape[3]))
        inequalities = np.zeros(self.valid_matches.sum())

        cix = 0
        for i in range(self.I):
            for j in range(self.J):
                for t in range(self.T):
                    if self.valid_matches[i][j][t]:
                        constraint_matrix[cix,:] = constraint_map[i,j,t,:] 
                        inequalities[cix] = self.pairing_weights[i,j,t]
                        cix += 1

        return constraint_matrix, inequalities


    def primal_solutions(self):
        m = Model("dynamicmatch_primal")
        m.modelSense = GRB.MAXIMIZE
        m.setParam( 'OutputFlag', False )

        weights = self.pairing_weights.reshape(self.pairing_weights.shape[0] * self.pairing_weights.shape[1] * self.pairing_weights.shape[2])
        c = -1 * self.pairing_weights.reshape(self.pairing_weights.shape[0] * self.pairing_weights.shape[1] * self.pairing_weights.shape[2])
        constraints = self.primal_constraint_matrix()

        keys = range(constraints.shape[1])
        variables = m.addVars(keys,
                        vtype=GRB.CONTINUOUS,
                         obj=weights,
                         name="primal",
                         lb=0)

        for constraint in constraints:
            m.addConstr(sum(variables[o]*c for o,c in filter(self.in_constraint, zip(variables,constraint))) <= 1)

        m.optimize()
        allocations = np.array([variables[var].X for var in variables]).reshape(self.pairing_weights.shape)

        return m.objVal, allocations


    def dual_solutions(self):
        md = Model("dynamicmatch_dual")
        md.modelSense = GRB.MINIMIZE
        md.setParam( 'OutputFlag', False )

        c_d = np.ones(self.valid_matches.sum())
        constraints_d, inequalities = self.dual_constraint_matrix()

        keys = range(constraints_d.shape[1])
        variables = md.addVars(keys,
                        vtype=GRB.CONTINUOUS,
                        obj=c_d,
                        name="dual",
                        lb=0)

        for cix, constraint in enumerate(constraints_d):
            con = sum(variables[o]*c for o,c in filter(self.in_constraint, zip(variables,constraint))) >= inequalities[cix]
            md.addConstr(sum(variables[o]*c for o,c in filter(self.in_constraint, zip(variables,constraint))) >= inequalities[cix])

        md.optimize()
        duals = np.array([variables[var].X for var in variables])
        betas = duals[duals.size - self.J:]
        alphas = duals[:duals.size - self.J]

        return md.objVal, alphas, betas
    
    def online_matching(self, alphas,betas,epsilon):

        alpha_map = self.make_alpha_mapping(alphas)
        online_allocations = np.zeros(self.valid_matches.shape)
        utility = 0
        candidate_matches = self.valid_matches.copy()

        for t in range(self.T):
            for i in range(self.I):
                for j in range(self.J):
                    if candidate_matches[i,j,t]:
                        asum = np.sum(alphas[alpha_map[i,j,t] == 1])

                        #allocate if less than epsilon
                        if np.abs(self.pairing_weights[i,j,t] - asum - betas[j]) <= epsilon:
                            online_allocations[i,j,t] = 1
                            utility += self.pairing_weights[i,j,t]

                            #prevent matches with resource during time period
                            candidate_matches[i,:,t:t+self.use_times[i]] = 0
                            candidate_matches[:,j,:] = 0

        return utility, online_allocations
    
    def solve(self,new_scenario=False):
        
        if new_scenario:
            self.create_tableau()
        
        objp, allocs = self.primal_solutions()
        objd, alphas, betas = self.dual_solutions()
        objo, online_allocs = self.online_matching(alphas,betas, 0)

        print("primal utility:",objp)
        print("dual utility:",objd)
        print("online utility:",objo)
        
        self.validate_allocation(allocs)
        self.validate_allocation(online_allocs)
            