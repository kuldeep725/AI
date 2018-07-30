# Name 		: Kuldeep Singh Bhandari
# Roll No.  : 111601009
# Aim  		: With environment "sea", with agent "bunny", help bunny to reach to the shore
#
class Environment :

	def __init__(self, coord, shore) :
		self.coord = coord
		self.shore = shore

	def updateState(self, move) :
		self.coord += move

	def providePerception(self) :
		return self.coord == self.shore

class Agent :

	def __init__(self) :
		self.move = 0

	def getPerception(self, envObj) :
		return envObj.providePerception()

	def takeAction(self, envObj) :
		if self.move == 0 :
			self.move += 1

		elif self.move < 0 :
			self.move = -(self.move - 1)

		else :
			self.move = -(self.move + 1)

		envObj.updateState(self.move)

[currentPos, shorePos] = list(map(int, input().split()))

agentObj = Agent()
envObj = Environment(currentPos, shorePos)
print("current position = %s " % str(envObj.coord))
print("shore position = %s " % str(envObj.shore))

print("Current Location, Bunny's Location, Perception, Action, Shore Location")
print(str(envObj.coord) + ", " + str(agentObj.move) + ", " + str(agentObj.getPerception(envObj)) + ", " + "None, " + str(envObj.shore))


while(agentObj.getPerception(envObj) == False) :
	
	x = agentObj.getPerception(envObj)
	agentObj.takeAction(envObj)
	print(str(envObj.coord) + ", " + str(agentObj.move) + ", " + str(agentObj.getPerception(envObj)) + ", " + str("right, "*(agentObj.move > 0)) + str("left, "*(agentObj.move < 0)) + str(envObj.shore))

print("Bunny reached Shore")