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
# print("current position = ", envObj.coord)
# print("shore position = ", envObj.shore)


print("Current Location, Bunny's Location, Perception, Action, Shore Location")
print(envObj.coord ,",", agentObj.move, ",", agentObj.getPerception(envObj), ", None,", envObj.shore)


while(agentObj.getPerception(envObj) == False) :
	
	x = agentObj.getPerception(envObj)
	agentObj.takeAction(envObj)
	print(envObj.coord, ",", (envObj.coord-currentPos) , ",", agentObj.getPerception(envObj), ",", "right,"*(agentObj.move > 0) + "left,"*(agentObj.move < 0), envObj.shore)

print("Bunny reached Shore")