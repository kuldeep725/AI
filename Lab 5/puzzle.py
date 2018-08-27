# Name         : Kuldeep Singh Bhandari
# Roll No.  : 111601009

import numpy as np
from Queue import Queue
import copy

class Node :
    # <M> is matrix
    # <pos> is position of empty block
    def __init__(self, M, pos, action) :
        self.M = M
        self.pos = pos
        self.action = action

class Coordinate :

    def __init__(self, x, y) :
        self.r = x
        self.c = y

def I (exp) :
    return int(exp)

def mod(a, b) :
    return a % b;

def d(coord, n) :
    return (n-coord.x) + (n-coord.y)

def parity(coord, M, n) :
    sum = 0
    for i in range(n*n) :
        if(M[i/4][i%4] == n*n) : continue
        for j in range(i+1, n*n) :
            sum += I(M[i/4][i%4] > M[j/4][j%4])
    return mod(d(coord) + sum, 2)

class Matrix :

    def __init__(self, M) :
        self.M = M

    def __eq__(self, other) :
        n = other.M.shape[0]
        for i in range(n) :
            for j in range(n) :
                if(self.M[i, j] != other.M[i,j]) :
                    return False
        return True

    def __hash__(self) :
        return hash(str(self.M))

class Environment :

    def __init__(self, n, M) :
        self.coord = Coordinate(n-1, n-1);
        self.n = n
        self.M = M
        
    def updateState (self, action) :
        print("updateState fired...")
        r = self.coord.r               #row number of empty block
        c = self.coord.c               #column number of empty block
        if(action == 'left') :         #move empty block left 
            if(c <= 0) : return False        #if action is invalid
            self.M[r][c], self.M[r][c-1] = (
                    self.M[r][c-1], self.M[r][c])
            self.coord = Coordinate(r, c-1);   #update position of empty block
            return True
        elif (action == 'right') :     #move empty block right
            if(c >= self.n-1) : return False   #if action is invalid
            self.M[r][c], self.M[r][c+1] = (
                    self.M[r][c+1], self.M[r][c])
            self.coord = Coordinate(r, c+1);    #update position of empty block
            return True
        elif (action == 'up') :        #move empty block up
            if(r <= 0) : return  False      #if action is invalid
            self.M[r-1][c], self.M[r][c] = (
                    self.M[r][c], self.M[r-1][c])
            self.coord = Coordinate(r-1, c);   #update position of empty block
            return True
        elif (action == 'down') :         #move empty block down
            if(r >= self.n-1) : return False      #if action is invalid
            self.M[r+1][c], self.M[r][c] = (
                    self.M[r][c], self.M[r+1][c])
            self.coord = Coordinate(r+1, c);  #update position of empty block
            return True
        else : return False
        
    def providePerception(self) :         #check if agent has reached goal
        for i in range(0, self.n) :
            for j in range(0, self.n) :
                if(i*self.n+j+1 != self.M[i][j]) : return False
        return True
    
class Agent :
    
    def takeAction(self, envObj, action) :
        return envObj.updateState(action)

    def getPerception(self, envObj) :
        return envObj.providePerception()

s = set()
q = Queue()
n = 3
# M = np.array([[3, 1], [2, 4]])
M = np.array([[1, 2, 3], [4, 5, 8], [6, 7, 9]]) # elements from 1 to n*n
envObj = Environment(n, M)
agent = Agent()
actionList = ['left', 'right', 'up', 'down']
q.put(Node(M, Coordinate(2, 2), 'None'))        # rows and columns from 0 to n-1
s.add(Matrix(M))
check = False
while(not q.empty()) :
    curr = q.get()
    print(curr.M)
    envObj.M = curr.M.copy()
    envObj.coord = copy.copy(curr.pos)
    if(agent.getPerception(envObj)) :
        check = True
        print("inside", curr.M)
        break

    for action in actionList :
        envObj.M = curr.M.copy()
        envObj.coord = copy.copy(curr.pos)
        flag = agent.takeAction(envObj, action)
        m = Matrix(copy.copy(envObj.M))
        print("flag", flag, "visited", m not in s)
        if(flag and (m not in s)) :
            q.put(Node(envObj.M.copy(), copy.copy(envObj.coord), action)) 
            s.add(m)

if(not check) :
    print("Odd parity")