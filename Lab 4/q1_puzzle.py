# Name         : Kuldeep Singh Bhandari
# Roll No.  : 111601009

class Coordinate :

	def __init__(self, x, y) :
		self.x = x
		self.y = y

def I (exp) :
	return int(exp)

def mod(a, b) :
	return a % b;

def d(coord) :
	return (n-coord.x) + (n-coord.y)

def parity(coord, M) :

	sum = 0
	for i in range(n*n) :
		if(M[i/4][i%4] == n*n) : continue
		for j in range(i+1, n*n) :
			sum += I(M[i/4][i%4] > M[j/4][j%4])
	return mod(d(coord) + sum, 2)

class Environment :

	def takeAction(self, action) :
		
