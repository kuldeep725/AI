# Name 		: Kuldeep Singh Bhandari
# Roll No.  : 111601009
import numpy as np
import sys

# to load Road text file
def loadRoad() :
    print("Loading road...")
#    return pickle.load(open('road', 'r'))
    return np.array(np.loadtxt('road.txt'))

# to load Vehicle text file
def loadVehicle() :
    print("Loading vehicle...")
#    return pickle.load(open('vehicle', 'r'))
    return np.array(np.loadtxt('vehicle.txt'))

# to load Time text file
def loadTime() :
    print("Loading time...")
    return np.array(np.loadtxt('time.txt'))

# to calculate speed of a vehicle
# parameter <x> dentoes number of vehicles ahead of current vehicle
def calcSpeed(x) :
    return np.exp(0.5*x)/(1 + np.exp(0.5*x)) + 15/(1 + np.exp(0.5*x))

# Environment class which possess informations about Roads, Vehicles,
# Vehicle's starting departure time, etc
class Environment :

    def __init__ (self) :
        self.data = 0
        # <R> stores road adjacent matrix
        self.R = loadRoad().astype(int)
        # <V> stores the vehicle path
        self.V = loadVehicle().astype(int)
        # <T> stores the departure time of vehicles from source
        self.T = loadTime()
        # converting time from minutes to hours
        self.T = self.T/60
        # manager <M> to choose the appropiate vehicle 
        # first column stores the time at which vehicle will move
        # ith row corresponds to ith vehicle
        # second column stores the current state of the vehicle 
        # second column can contain values [0, 1, 2, 3, 4]
        # which represents the current position of the vehicle in the
        # path from source to destination which includes five nodes
        self.M = np.zeros((self.V.shape[0], 2))   
        # setting first column of M to be the departure time of vehicles
        # from source
        self.M[:, 0] = self.T

# creating object of class Environment
env = Environment()

# roadList is a list of list of lists
# roadList[i][j] gives a list containing endtimes of all the vehicles
# which passed through road (i, j)
roadList = [[[] for i in range(env.R.shape[0])] for j in range(env.R.shape[0])]

# <TOT_RUN> stores the number of times manager needs to choose a vehicle
TOT_RUN = env.V.shape[0] * (env.V.shape[1] - 1)
# <out> stores
out = np.zeros((env.V.shape[0], env.V.shape[1]+1))
out[:, 0] = [ int(x) for x in list(range(env.V.shape[0])) ]
out[:, 1] = env.T
for i in range(0, TOT_RUN) :
    
    x = np.where(env.M[:, 0] == np.min(env.M[:, 0]))[0][0]
    cnt = 0 
    startTime = env.M[x, 0]
    y = int(env.M[x, 1])
    
    ind1 = env.V[x, y]
    ind2= env.V[x, y+1]
    
    for time in roadList[ind1][ind2] :
        if(time > startTime) :
            cnt += 1
            
    endTime = startTime + env.R[ind1, ind2] / calcSpeed(cnt)
    out[x][y+2] = endTime
    env.M[x, 0] = endTime
    env.M[x, 1] += 1
    roadList[ind1][ind2].append(endTime)
    
    if(int(env.M[x, 1]) == env.V.shape[1]-1) :
        env.M[x, 0] = sys.maxsize

print(out)
np.savetxt('output.csv', np.asarray(out), '%5.10f', delimiter=',', 
           header='Vehicle, site1,site2,site3,site4,site5')


