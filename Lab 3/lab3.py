# Name         : Kuldeep Singh Bhandari
# Roll No.  : 111601009

import random
import math
class Coordinate :
    
#    def __init__(self, x) :
#        self.x = x;
#        self.y = 0;
#        self.z = 0;
#        
#    def __init__(self, x, y) :
#        self.x = x;
#        self.y = y;
#        self.z = 0;
        
    def __init__(self, x, y = 0, z = 0) :
        self.x = x;
        self.y = y;
        self.z = z;
        
#    def equalTo(self, coordGoal) :
#        return (self.x == coordGoal.x and 
#            self.y == coordGoal.y and
#            self.z == coordGoal.z)

class Environment :

    def __init__(self, curr_dim, L) :
        self.curr_dim = curr_dim; 
        self.L = L;
        
    def initialise(self) :
        if(self.curr_dim == 1) :
            self.coord = Coordinate(math.ceil(self.L/2));
        elif(self.curr_dim == 2) :
            self.coord = Coordinate(math.ceil(self.L/2), math.ceil(self.L/2));
        elif(self.curr_dim == 3) :
            self.coord = Coordinate(math.ceil(self.L/2), math.ceil(self.L/2),
                                            math.ceil( self.L/2));
            
    def updateState(self, otherCoord, action) :
        if(action == 'left') :
            if(self.coord.x <= 0) : return False;
            self.coord.x += -1
        elif(action == 'right') :
            if(self.coord.x >= L) :return False;
            self.coord.x += 1
        elif(action == 'up') :
            if(self.coord.y >= L) : return False;
            self.coord.y += 1;
        elif(action == 'down') :
            if(self.coord.y <= 0) : return False;
            self.coord.y += -1;
        elif(action == 'forward') :
            if(self.coord.z >= L) : return False;
            self.coord.z += 1;
        elif(action == 'backward') :
            if(self.coord.z <= 0) : return False;
            self.coord.z += -1;
#        self.coord.x = otherCoord.x + math.ceil(L/2);
#        if(self.curr_dim >= 2) : self.coord.y = otherCoord.y + math.ceil(L/2);
#        if(self.curr_dim >= 3) : self.coord.z = otherCoord.z + math.ceil(L/2);
        return True;
    
    def providePerception(self) :
        return (self.coord.x + self.coord.y + self.coord.z) == self.curr_dim * self.L
    
    def toString(self) :
        return ('('+ str(int(self.coord.x)) + ', ' + str(int(self.coord.y)) + ', '+
                str(int(self.coord.z)) + ')')
    
    
class Agent :
    
    def __init__(self) :
        self.coord = Coordinate(0);
        
    def getPerception(self, envObj) :
        return envObj.providePerception()
        
    def takeAction(self, envObj, curr_dim, L) :
        actionList = ['left', 'right', 'up', 'down', 'forward', 'backward']
        self.action = actionList[random.randint(0, 2*curr_dim-1)]
        status = envObj.updateState(self.coord, self.action);
        if(not status) : return;
        
        if(self.action == 'left') :
#            if(self.coord.x <= -L/2) : return;
            self.coord.x += -1
        elif(self.action == 'right') :
#            if(self.coord.x >= L/2) :return;
            self.coord.x += 1
        elif(self.action == 'up') :
#            if(self.coord.y >= L/2) : return;
            self.coord.y += 1;
        elif(self.action == 'down') :
#            if(self.coord.y <= -L/2) : return;
            self.coord.y += -1;
        elif(self.action == 'forward') :
#            if(self.coord.z >= L/2) : return;
            self.coord.z += 1;
        elif(self.action == 'backward') :
#            if(self.coord.z <= -L/2) : return;
            self.coord.z += -1;
            
        
    
    def toString(self) :
        return ('('+ str(int(self.coord.x)) + ', ' + str(int(self.coord.y)) + ', '+
                str(int(self.coord.z)) + ')')
        
while True :
    
    print("====================================================")
    dim = random.randint(1, 3);
    print("Dimension =", dim);
    L = float(input("Enter positive value of L : "))
    if(L < 0) : break;
    envObj = Environment(dim, L);
    envObj.initialise()
    agent = Agent();
    print("Current Location, Bunny's Location, Action")
    print(envObj.toString(), agent.toString(), "None");
    while(not agent.getPerception(envObj)) :
        agent.takeAction(envObj, dim, L);
        print(envObj.toString(), agent.toString(), agent.action);
    
    choice = input("Do you want to continue (y or n) : ");
    if(choice == 'n' or choice == 'N') : break;
        
        