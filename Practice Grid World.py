
# coding: utf-8

# # Grid World

# In[1]:


import numpy as np
import sys


# In[16]:


class Environment :
    def __init__(self, R, P_actionSuccess, gamma) :
        self.R = R
        self.m = R.shape[0]
        self.n = R.shape[1]
        self.gamma = gamma
        self.P_actionSuccess = P_actionSuccess
        
        # initialize value and policy
        self.V = np.zeros(R.shape)
        self.policy = np.array(np.empty(R.shape), dtype=np.str)
#         self.V = np.zeros(R.shape)
#         self.policy = np.array([["right", "right", "right", "right"]
#                         ,["right", "right", "right", "right"]
#                         ,["right", "right", "right", "right"]
#                         ,["right", "right", "right", "right"]])
        self.policy[:] = "left"
        self.actionList = ["left", "right", "up", "down"]

    def getSuccessor(self, i, j, action) :
        if(action == "left" and j > 0) :
            return self.V[i][j-1]
        elif(action == "right" and j < self.n-1)  :
            return self.V[i][j+1]
        elif(action == "up" and i > 0) :
            return self.V[i-1][j]
        elif(action == "down" and i < self.m-1) :
            return self.V[i+1][j]
        return None
  
    def Bellmann(self, i , j) :
        bestAction = "None"
        bestValue = -sys.maxsize
        reward = self.R[i, j]
        for action in self.actionList :
            val = 0
            for k, p in enumerate(self.P_actionSuccess[action]) :
                V_successor = self.getSuccessor(i, j, self.actionList[k])
                if(V_successor is not None) :
                    val += (p * (reward + (gamma*V_successor)))
#                     val += p*-1
                else :
                    V_successor = self.V[i, j]
                    val += p * (-1 + (gamma*V_successor))
                
            if(val > bestValue) :
                bestValue = val
                bestAction = action

        return bestValue, bestAction
        
    def ValueIteration(self) :
        iter = 0
        MAX_ITER = 10000
        print(50*'-')
        print(' '*15, "Value Iteration ")
        print(50*'-')
        while(iter < MAX_ITER) :
            iter += 1
            delta = 0
            for i in range(self.m) :
                for j in range(self.n) :
                    oldV = self.V[i, j]
                    self.V[i, j], self.policy[i, j] = self.Bellmann(i, j)
                    delta = max(delta, abs(oldV - self.V[i, j]))
#             print(self.V)
#             print("delta = ", delta)
            if(delta < 0.01) :
                break
                
        if(iter == MAX_ITER) :
            print("\nValue iteration did not converge in", iter, "iterations")
        else :
            print("\nValue Iteration converges in", iter, "steps")
            print("\nOptimal Value : \n", self.V)
            print("\nOptimal Policy : \n", self.policy)
        print(50*'-')
        
    def findNewPolicy(self) :
        iter = 0
        MAX_ITER = 10000
        # Policy Evaluation
        while(iter < MAX_ITER) :
            iter += 1
            delta = 0
            for i in range(self.m) :
                for j in range(self.n) :
                    prevVal = self.V[i, j]
                    reward = self.R[i, j]
                    action = self.policy[i, j]
                    val = 0
                    for k, p in enumerate(self.P_actionSuccess[action]) :
                        V_successor = self.getSuccessor(i, j, self.actionList[k])
                        if(V_successor is not None) :
                            val += (p * (reward + (gamma*V_successor)))
                        else :
                            V_successor = self.V[i, j]
                            val += (p * (-1 + (gamma * V_successor)))
                    delta = max(delta, abs(val-prevVal))
                    self.V[i, j] = val
#             print("Value : ", self.V)

            if(delta < 0.01) :
                break
#         print("Value : ", self.V)
        # Policy Improvement
        for i in range(self.m) :
            for j in range(self.n) :
                reward = self.R[i, j]
                max_val = -1e7
                bestAction = self.policy[i, j]
                for action in self.actionList :
                    val = 0
                    for k, p in enumerate(self.P_actionSuccess[action]) :
                        V_successor = self.getSuccessor(i, j, self.actionList[k])
                        if(V_successor is not None) :
                            val += (p * (reward + gamma * V_successor))
                        else :
                            V_successor = self.V[i, j]
                            val += (p * (-1 + gamma * V_successor))
                    if(max_val < val) :
                        bestAction = action
                        max_val = val
                self.policy[i, j] = bestAction
                    
    def policyIteration(self) :
        iter = 0
        MAX_ITER = 10000
        while(iter < MAX_ITER) :
            iter += 1
            prevPolicy = self.policy.copy()
            self.findNewPolicy()
            if(np.array_equal(prevPolicy, self.policy)) :
                break
                
        if(iter == MAX_ITER) :
            print("\nPolicy iteration did not converge in", iter, "iterations")
        else :
            print("\nPolicy Iteration converges in", iter, "steps")
            print("\nOptimal Value : \n", self.V)
            print("\nOptimal Policy : \n", self.policy)
        print(50*'-')


# # Driver Program 

# In[17]:


P_actionSuccess = { "left" : [0.8, 0, 0.1, 0.1], "right" : [0, 0.8, 0.1, 0.1], 
                                 "up" : [0.1, 0.1, 0.8, 0], "down" : [0.1, 0.1, 0, 0.8] }
R = np.array([[0, 0.45, 1, 0.9]
               ,[0.23, 1.25, 0, 0]
               ,[0, 0.45, 0.75, 0]
               ,[0.85, 1.5, 2.5, 0.85]])
gamma = 0.98
envObj1 = Environment(R, P_actionSuccess, gamma)
envObj2 = Environment(R, P_actionSuccess, gamma)
print("-"*15, "REWARD ", "-"*15)
print(R)
print("-"*39)
envObj1.ValueIteration()
envObj2.policyIteration()

