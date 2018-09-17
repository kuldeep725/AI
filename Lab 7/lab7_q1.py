# Member : (1) Kuldeep Singh Bhandari
#		   (2) Amit Vikram Singh
from queue import PriorityQueue 
import numpy as np
import math
import copy

INF = int(1e7)

class Coordinate :
	def __init__(self, x, y) :
		self.x = x
		self.y = y

	def __repr__(self) :
		return "(" + str(int(self.x)) + ", " + str(int(self.y)) + ")"

class QueueNode :
	def __init__(self, v, p, g, h) :
		self.v = v 		# vertex
		self.p = p		# parent
		self.g = g		# total cost till now
		self.h = h		# cost to go

	def __lt__(self, other) :
		return self.g + self.h < other.g + other.h
class Node :
	def __init__(self, v, w) :
		self.v = v 		# vertex
		self.w = w 		# weight

class GraphNode :
	def __init__ (self, adj=[], g = INF, h = 0, visited = False, pos = None) :
		self.pos = pos
		self.adj = adj
		self.h = h
		self.visited = visited

class Environment :

	def __init__(self, source, goal, n):
		self.source = source
		self.goal = goal
		self.n = n
		self.graph = [ GraphNode([]) for i in range(n+1)]
		
	def initializeGraph(self, Nodes, Distances) :
		goalCoordinate = Coordinate(Nodes[self.goal-1][1], Nodes[self.goal-1][2])
		n = self.n
		for i in range(1, n+1) :
			self.graph[i].pos = Coordinate(Nodes[i-1][1], Nodes[i-1][2])
			self.graph[i].h = self.getDistance(self.graph[i].pos, goalCoordinate)
			for j in range(1, n+1) :
				if(Distances[i-1][j] != 'N') :
					self.graph[i].adj.append(Node(j, float(Distances[i-1][j])))

	def getDistance(self, pos, goal) :
		return math.sqrt((pos.x-goal.x)**2 + (pos.y-goal.y)**2)

def printPath (envObj, node) :
	if(node is None) : 
		return
	printPath(envObj, node.p)
	print(envObj.graph[node.v].pos, end=" ")

Nodes = np.loadtxt("nodes.csv", delimiter = ",", skiprows=1)
Distances = np.loadtxt("distance.csv", dtype="str", delimiter = ",", skiprows=1)
n = Nodes.shape[0]
q = PriorityQueue()
source = 1
goal = 14
envObj = Environment(source, goal, n)
envObj.initializeGraph(Nodes, Distances)

destinationNode = None
q.put(QueueNode(source, None, 0, 0))
envObj.graph[source].g = 0;

while(not q.empty()) :

	curr = q.get()
	if(curr.v == goal) :
		destinationNode = curr
		break

	envObj.graph[curr.v].visited = True;

	for adjNode in envObj.graph[curr.v].adj :
		if(not envObj.graph[adjNode.v].visited) :
			q.put(QueueNode(adjNode.v, curr, copy.copy(curr.g + adjNode.w), envObj.graph[adjNode.v].h))

if(destinationNode is not None) :
	print ("goal reached")
	print ("Path : ", end = " ")
	printPath(envObj, destinationNode)
	print()
else :
	print("Not reached")