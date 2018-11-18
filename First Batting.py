
# coding: utf-8

# ### Lab 09
# #### Problem: Batting First
# ##### Members:
#         1. Amit Vikram Singh(111601001)
#         2. Kuldeep Singh Bhandari(111601009
# #### Logic :
# > We are applying bottom-up approach to find the optimal value and optimal policy for batting first problem. This is **Value Iteration** technique using Dynamic Programming (here we are storing answer to sub-problems in tabular fashion). We are beginning with **"0 balls left and {0-10} wickets left"** scenario and building up using previous results to find the answer for **"300 balls left and 10 wickets left"** scenario.

# In[9]:


import numpy as np
import sys


# ## Define State

# In[8]:


class State :
    def __init__(self, balls_left, wickets_left) :
        self.balls_left = balls_left
        self.wickets_left = wickets_left
        
class Environment :
    def __init__(self, pr_min, pr_max, pw_min, pw_max) :
        self.pr_min = pr_min
        self.pr_max = pr_max
        self.pw_min = pw_min
        self.pw_max = pw_max
        self.TOT_BALLS = 301
        self.V = np.zeros((self.TOT_BALLS, 11))    # value function
        self.policy = np.zeros((self.TOT_BALLS, 11))   # policy function
        self.actionList = [1, 2, 3, 4, 6]
        self.Prob_out = [ {self.actionList[j] : self.getWicketProb(i, j) for j in range(5)} for i in range(10) ]
        self.Prob_run = [ self.getRunProb(i) for i in range(10) ]
        
    def getWicketProb (self, i, j) :
        return self.pw_max[j] + ((self.pw_min[j] - self.pw_max[j])*((i-1)/9))
        
    def getRunProb (self, x) :
        return self.pr_min + ((self.pr_max - self.pr_min)*((x-1)/9))
    
    def Bellmann (self, i, j) :
        max_val = 0
        bestAction = 0
        for action in self.actionList :
            p_out = self.Prob_out[j-1][action]
            p_run = self.Prob_run[j-1]
            val = (1 - p_out)*(p_run * action + self.V[i-1][j]) + (p_out * self.V[i-1][j-1])
            if(val > max_val) :
                max_val = val
                bestAction = action
        return max_val, bestAction
        
    def ValueIteration(self) :
        for i in range(1, self.TOT_BALLS) :
            for j in range(1, 11) :
                self.V[i][j], self.policy[i][j] = self.Bellmann (i, j)


# ## Driver Code

# In[10]:


pw_min = [0.01, 0.02, 0.03, 0.1, 0.3]
pw_max = [0.1, 0.2, 0.3, 0.5, 0.7]
envObj = Environment(0.5, 0.8, pw_min, pw_max)
envObj.ValueIteration()
print("Saving Values in file Value.txt...")
print("Saving Policies in file Policy.txt...")
np.savetxt("Value.txt", envObj.V, fmt="%5.2f")
np.savetxt("Policy.txt", envObj.policy, fmt="%d")

