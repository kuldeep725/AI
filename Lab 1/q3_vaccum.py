# Name 		: Kuldeep Singh Bhandari
# Roll No.  : 111601009

class Environment :

	def __init__(self, dx, dy, sx, sy) :
		self.dx = dx
		self.dy = dy
		self.sx = sx
		self.sy = sy

	def updateState(self, movex, movey) :
		self.sx += movex
		self.sy += movey

	def providePerception(self) :
		return self.sx == self.dx and self.sy == self.dy

class Agent :

	def __init__(self) :
		self.movex = 0
		self.movey = 0

	def getPerception(self) :
		return providePerception(self)

	def takeAction(self) :
		if(movex == 0 && movey == 0) :
			movex 