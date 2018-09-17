# Name     : Kuldeep Singh Bhandari
# Roll No. : 111601009
# Idea     : 
            # First of all, we can assume 1st blank position to be (n*n-1) and second blank position to be (n*n) then the idea is similar to that of (n*n-1) puzzle
            # In this case, we need to take steps using both blanks (instead of one blank as in case of n*n-1 puzzle) using heuristic "number of inversion count" 
            # that is how far our matrix is from being sorted.
# Input type :
    # keep "input.txt" file in the same folder as that of this file
    # In the file, just write matrix in which the elements are seperated by comma
    # Sample Input :
        # 8, 11, 15, 16
        # 6, 10, 9, 12
        # 5, 3, 13, 7
        # 14, 4, 1, 2
from __future__ import print_function
import numpy as np
from Queue import Queue
from Queue import PriorityQueue
import copy
import math

# class <Node> represents a state which possess informations like
# <M> which is matrix of the current state,
# <pos1> which is the coordinate of current position of 1st empty block,
# <pos2> which is the coordinate of current position of 2nd empty block,
# <action> which is action taken by parent to reach this state,
# <parent> which is the parent state of this state,
# <blank> which is the blank type using which this node is achieved
# <h> which is heuristic for the model

class Node :
    def __init__(self, M, pos1, pos2, action, parent, blank, h) :
        self.M = M
        self.pos1 = pos1                   # type : <Coordinate> class
        self.pos2 = pos2
        self.action = action
        self.parent = parent             # type : <Node> class
        self.blank = blank
        self.h = h

    def __cmp__(self, other) :
        return cmp(self.h, other.h)

# class <Coordinate> represents the x-coordinate and y-coordinate of a block
class Coordinate :

    def __init__(self, x, y) :
        self.r = x
        self.c = y

    def __repr__(self) :
        return "("+ str(self.r) + ", " + str(self.c) + ")"

# class <Matrix> which is useful for keeping track of explored states
class Matrix :

    def __init__(self, M) :
        self.M = M

    # function <__eq__> overrides "==" operator so whenever we write 
    # "matrix1 == matrx2", it will check whether all corresponding elements
    # of both matrices are equal
    def __eq__(self, other) :
        n = other.M.shape[0]
        for i in range(n) :
            for j in range(n) :
                if(self.M[i, j] != other.M[i,j]) :
                    return False
        return True

    # overriding hashing function
    def __hash__(self) :
        return hash(str(self.M))

class Environment :

    def __init__(self, n, M, coord1, coord2) :
        self.coord1 = coord1        # coordinate of first blank
        self.coord2 = coord2        # coordinate of second blank
        self.M = M                  # matrix
        self.n = n                  # number of rows/columns of matrix

    def updateState (self, action, blank) :
#        print("updateState fired...")
        if(blank == "blank1") :
            r = self.coord1.r               #row number of empty block1
            c = self.coord1.c               #column number of empty block1
        else :
            r = self.coord2.r               #row number of empty block2
            c = self.coord2.c               #column number of empty block2

        if(action == 'left') :         #move empty block left 
            if(c <= 0) : return False        #if action is invalid
            self.M[r][c], self.M[r][c-1] = (
                    self.M[r][c-1], self.M[r][c])
            if(blank == "blank1") : 
                self.coord1 = Coordinate(r, c-1);
                if(self.isEqual(self.coord2, Coordinate(r, c))) :
                    self.coord2 = Coordinate(r, c)
            else :
                self.coord2 = Coordinate(r, c-1);   
                if(self.isEqual(self.coord1, Coordinate(r, c))) :
                    self.coord1 = Coordinate(r, c)
            return True
        elif (action == 'right') :     #move empty block right
            if(c >= self.n-1) : return False   #if action is invalid
            self.M[r][c], self.M[r][c+1] = (
                    self.M[r][c+1], self.M[r][c])
            if(blank == "blank1") : 
                self.coord1 = Coordinate(r, c+1);   
                if(self.isEqual(self.coord2, Coordinate(r, c))) :
                    self.coord2 = Coordinate(r, c)
            else :
                self.coord2 = Coordinate(r, c+1);   
                if(self.isEqual(self.coord1, Coordinate(r, c))) :
                    self.coord1 = Coordinate(r, c)
            return True
        elif (action == 'up') :        #move empty block up
            if(r <= 0) : return  False      #if action is invalid
            self.M[r-1][c], self.M[r][c] = (
                    self.M[r][c], self.M[r-1][c])
            if(blank == "blank1") : 
                self.coord1 = Coordinate(r-1, c); 
                if(self.isEqual(self.coord2, Coordinate(r, c))) :
                    self.coord2 = Coordinate(r, c)
                    print("coord2 = ", (r, c))
            else :
                self.coord2 = Coordinate(r-1, c);  
                if(self.isEqual(self.coord1, Coordinate(r, c))) :
                    self.coord1 = Coordinate(r, c)
                    print("coord1 = ", (r, c))
            return True
        elif (action == 'down') :         #move empty block down
            if(r >= self.n-1) : return False      #if action is invalid
            self.M[r+1][c], self.M[r][c] = (
                    self.M[r][c], self.M[r+1][c])
            if(blank == "blank1") : 
                self.coord1 = Coordinate(r+1, c); 
                if(self.isEqual(self.coord2, Coordinate(r, c))) :
                    self.coord2 = Coordinate(r, c)
            else :
                self.coord2 = Coordinate(r+1, c);  
                if(self.isEqual(self.coord1, Coordinate(r, c))) :
                    self.coord1 = Coordinate(r, c)
            return True
        else : return False
        
    def providePerception(self) :         #check if agent has reached goal
        for i in range(0, self.n) :
            for j in range(0, self.n) :
                if(i*self.n+j+1 != self.M[i,j]) : return False
        return True

    # check if both coordinates are equal
    def isEqual(self, c1, c2) :
        return (c1.r == c2.r) and (c1.c == c2.c)

class Agent :
    
    def takeAction(self, envObj, action, blank) :
        return envObj.updateState(action, blank)

    def getPerception(self, envObj) :
        return envObj.providePerception()

# for counting number of inversions for heuristic purpose
def countInversions (M, n) :
    invCount = 0
    for i in range(n*n) :
        p_row = i // n
        p_col = i % n
        act_row = (M[p_row][p_col]-1) // n
        act_col = (M[p_row][p_col]-1) % n 
        invCount += abs(act_row-p_row) + abs(act_col - p_col)
        # print(p_row, "==", p_col)
        # print(act_row, "--", act_col)
        # print(M[p_row][p_col], " : ", invCount)
    return invCount

# for finding coordinate of <num> in the matrix M
def getIndex(M, num, n) :
    for i in range(n) :
        for j in range(n) :
            if(M[i][j] == num) : return Coordinate(i, j)
    return None

# for printing path
def printPath(curr, tot) :
    if(curr is None) : return 0
    tot += 1+printPath(curr.parent, tot)
    print((curr.action, curr.blank, curr.h))
    print(curr.M)
    return tot

# stores explored states
s = set()
# q = Queue()
q = PriorityQueue()
# M = np.array([[3, 1, 2], [4, 5, 8], [6, 7, 9]]) # elements from 1 to n*n
# M = np.array([[3, 6, 2], [4, 8, 1], [5, 7, 9]]) # elements from 1 to n*n
# M = np.array([[8, 11, 15, 16], [6, 10, 9, 12], [5, 3, 13, 7], [14, 4, 1, 2]])
M = np.loadtxt("input.txt", delimiter=",").astype(int)
n = M.shape[0]
# print(countInversions(M, n))
blank1 = getIndex(M, n*n-1, n)          # coordinate of first blank
blank2 = getIndex(M, n*n, n)            # coordinate of second blank
envObj = Environment(n, M, blank1, blank2)
agent = Agent()
actionList = ['left', 'right', 'up', 'down']
b1 = "blank1"
b2 = "blank2"
q.put(Node(M.copy(), blank1, blank2, None, None, b1, countInversions(M, n)))
q.put(Node(M.copy(), blank1, blank2, None, None, b2, countInversions(M, n)))
print(envObj.coord1, envObj.coord2)
print(blank1, blank2)
s.add(Matrix(M.copy()))
while (not q.empty()) :
    curr = q.get()
    envObj.coord1 = copy.copy(curr.pos1)
    envObj.coord2 = copy.copy(curr.pos2)
    # update environment matrix with new matrix of state <curr>
    envObj.M = curr.M.copy()
    # if(curr.parent is not None) :
    #     print(curr.action, curr.blank, curr.h, "\n", curr.parent.M, "\n", curr.M)
    # else :
    #     print(curr.action, curr.blank, curr.h, "\n", curr.M)

    if(agent.getPerception(envObj)) :
        break

    for action in actionList :
        envObj.M = curr.M.copy()
        envObj.coord1 = copy.copy(curr.pos1)
        envObj.coord2 = copy.copy(curr.pos2)

        flag = agent.takeAction(envObj, action, b1)
        m = Matrix(copy.copy(envObj.M))
#        print("flag", flag, "visited", m not in s)
        if(flag and (m not in s)) :
            q.put(Node(envObj.M.copy(), copy.copy(envObj.coord1), copy.copy(envObj.coord2), action, curr, b1, countInversions(envObj.M, n))) 
            s.add(m)

        # update environment matrix with new matrix of state <curr>
        envObj.M = curr.M.copy()
        envObj.coord1 = copy.copy(curr.pos1)
        envObj.coord2 = copy.copy(curr.pos2)
        flag = agent.takeAction(envObj, action, b2)
        m = Matrix(copy.copy(envObj.M))
#        print("flag", flag, "visited", m not in s)
        if(flag and (m not in s)) :
            q.put(Node(envObj.M.copy(), copy.copy(envObj.coord1), copy.copy(envObj.coord2), action, curr, b2, countInversions(envObj.M, n))) 
            s.add(m)

print("path reached")
tot = printPath(curr, 0)
print("Total steps = ", tot)

