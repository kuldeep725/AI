
# coding: utf-8

# #### Mountain Car Problem

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


gridSize = (100, 100)
velocity = (-0.07, 0.07)
position = (-1.2, 0.6)
actionSet = [-1, 0, 1]
posStep = (position[1] - position[0])/(gridSize[0] - 1)
print(posStep)
velStep = (velocity[1] - velocity[0])/(gridSize[1] - 1)
print(velStep)

def updateFunction(oldPos, oldVel, action):
    newVel = oldVel + (action * 0.001) + np.cos(3*oldPos)*(-0.0025)
    newVel = min(max(newVel, velocity[0]), velocity[1])
    
    newPos = oldPos + newVel
    newPos = min(max(newPos, position[0]), position[1])
    
    if(newPos<= position[0]):
        newVel = 0
    
    return newPos, newVel

def normInfinity(currValue, optimalValue):
    maxDiff = 0
    for i in range(env.rowNum):
        for j in range(env.colNum):
            maxDiff = max(maxDiff, abs(currValue[i, j] - optimalValue[i, j]))
    return maxDiff

    
def reachedGoal(pos):
    if pos < 0.6:
        return False
    return True

def getIndex(pos, vel):
#     print(pos - position[0])
    posInd = (pos - position[0])/posStep
    velInd = (vel - velocity[0])/velStep
    
    posInd = np.ceil(posInd)
    velInd = np.ceil(velInd)
    
#     posInd = min(max(posInd, 0), gridSize[0] - 1)
#     velInd = min(max(velInd, 0), gridSize[1] - 1)
    
    return int(posInd), int(velInd) 

def getState(posInd, velInd):   #row, col
    pos = position[0] + posInd*posStep
    vel = velocity[0] + velInd*velStep
    
    return pos, vel


# - Value Iteration

# In[3]:


getIndex(-1.19, 0)


# In[4]:


minNum = -1000000
gamma = 1
def valueIteration(value, policy):
    for iter1 in range(500):
        if(iter1%50 == 0): print("Iteration: ", iter1)
        for posInd in range(gridSize[0]):
            for velInd in range(gridSize[1]):
                optimalVal = minNum
                optimalAction = -1
                for action in actionSet:
                    currVal = 0
                    pos, vel = getState(posInd, velInd)
                    newPos, newVel = updateFunction(pos, vel, action)
                    newPosInd, newVelInd = getIndex(newPos, newVel)
    
                    if(reachedGoal(newPos)):
                        currVal += 0 + value[newPosInd, newVelInd]
                    else:
                        currVal += -1 + value[newPosInd, newVelInd]

                    if(currVal > optimalVal):
                        optimalVal = currVal
                        optimalAction = action

                value[posInd, velInd] = optimalVal
                policy[posInd, velInd] = optimalAction


# In[5]:


value = np.zeros(gridSize)
policy = np.zeros(gridSize)
valueIteration(value, policy)


# In[6]:


plt.imshow(value, cmap = "hot" , interpolation="nearest")
plt.show()


# In[7]:


np.savetxt("value.txt", value, fmt = "%i")
np.savetxt("policy.txt", policy, fmt = "%i")

