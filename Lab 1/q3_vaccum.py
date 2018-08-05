# Name 		: Kuldeep Singh Bhandari
# Roll No.  : 111601009
# Aim		: How robot can collect the dirt

# class Environment begins
class Environment :

	def __init__(self, dx, dy, sx, sy) :
		self.dx = dx		# destination x coordinate
		self.dy = dy		# destination y coordinate
		self.sx = sx		# starting x coordinate
		self.sy = sy        # starting y coordinate
		self.movex = 0
		self.movey = 0

	def updateState(self, movex, movey) :
		self.movex = movex;
		self.movey = movey;

	def providePerception(self) :
		return (self.sx + self.movex == self.dx) and (self.sy + self.movey == self.dy)

# class Agent begins
class Agent :

	def getPerception(self, envObj) :
		return envObj.providePerception()

	def takeAction(self, envObj, movex, movey) :
		envObj.updateState(movex, movey)
		

def callAgent(agentObj, envObj, i, j, action) :

	if(agentObj.getPerception(envObj)) :
		return True

	agentObj.takeAction(envObj, i, j)
	print("(", (envObj.sx + envObj.movex), ",", (envObj.sy + envObj.movey), "), ", "\t(", i, ",", j,
	 "),\t", agentObj.getPerception(envObj), ",\t", agentObj.getPerception(envObj)* "\tNone", 
	 (not agentObj.getPerception(envObj))*action, ",\t", "(", envObj.dx, ",", envObj.dy, ")")

	return False

# main program begins here

agentObj = Agent()
[sx, sy, dx, dy] = list(map(int, input().split()))
envObj = Environment(dx, dy, sx, sy)

print("Current Location, Robot Location, Perception, Action, Dirt Location")
print("(", envObj.sx, ",", envObj.sy, "), ", "\t(", 0, ",", 0,
	 "),\t", agentObj.getPerception(envObj), ",\t", "right", 
	 ",\t", "(", envObj.dx, ",", envObj.dy, ")")

c = 1
flag = False
while True :

	callAgent(agentObj, envObj, c, 0, "up")

	for i in range(1, c) :
		flag = callAgent(agentObj, envObj, c, i, "up")
		if(flag) : break;

	for i in range(c, -c, -1): 
		flag = callAgent(agentObj, envObj, i, c, "left")
		if(flag) : break
	for i in range(c, -c, -1) :
		flag = callAgent(agentObj, envObj, -c, i, "down")
		if(flag) : break
	for i in range(-c, c) :
		flag = callAgent(agentObj, envObj, i, -c, "right")
		if(flag) : break
	for i in range(-c, 1) :
		flag = callAgent(agentObj, envObj, c, i, "right" if (i == 0) else "up")
		if(flag) : break

	c += 1
	if(flag) : break

print("Dirt found")