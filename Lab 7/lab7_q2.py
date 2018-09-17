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
	def __init__(self, v, p, g, h, vehicle, budget) :
		self.v = v 		# vertex
		self.p = p		# parent
		self.g = g		# total cost till now
		self.h = h		# cost to go
		self.vehicle = vehicle
		self.budget = budget

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

	def __init__(self, source, goal, n, cycleSpeed, busSpeed, MAX_BUS_SPEED):
		self.source = source
		self.goal = goal
		self.n = n
		self.graph = [ GraphNode([]) for i in range(n+1)]
		self.cycleSpeed = cycleSpeed
		self.busSpeed = busSpeed
		self.MAX_BUS_SPEED = MAX_BUS_SPEED
		
	def initializeGraph(self, Nodes, Distances) :
		goalCoordinate = Coordinate(Nodes[self.goal-1][1], Nodes[self.goal-1][2])
		n = self.n
		for i in range(1, n+1) :
			self.graph[i].pos = Coordinate(Nodes[i-1][1], Nodes[i-1][2])
			for j in range(1, n+1) :
				if(Distances[i-1][j] != 'N') :
					self.graph[i].adj.append(Node(j, float(Distances[i-1][j])))

	def findHeuristic(self) :
		q = PriorityQueue()
		n = self.n
		exploredList = [False for i in range(n+1)]
		q.put(self.source)
		time = [INF for i in range(n+1)]
		time[self.source] = 0

		while(not q.empty()) :
			curr = q.get()
			exploredList[curr] = True
			for adjNode in self.graph[curr].adj :
				if(not exploredList[adjNode.v]) :
					q.put(adjNode.v)
					if(adjNode.w <= 3) :
						time[adjNode.v] = time[curr] + (adjNode.w / self.cycleSpeed)
					else :
						time[adjNode.v] = time[curr] + (adjNode.w / self.MAX_BUS_SPEED)

		for i in range(1, n+1) :
			self.graph[i].h = time[i]

def printPath (envObj, node) :
	if(node is None) : 
		return
	printPath(envObj, node.p)
	print(envObj.graph[node.v].pos, ", ", node.vehicle, ", ", node.g, ", ", node.budget)

Nodes = np.loadtxt("nodes.csv", delimiter = ",", skiprows=1)
Distances = np.loadtxt("distance.csv", dtype="str", delimiter = ",", skiprows=1)
n = Nodes.shape[0]
q = PriorityQueue()
source = 1
goal = 14
MAX_BUS_SPEED = 50
cycleSpeed = 25
busSpeed = 50
totalBudget = 5
busFare = 5
envObj = Environment(source, goal, n, cycleSpeed, busSpeed, MAX_BUS_SPEED)
envObj.initializeGraph(Nodes, Distances)
envObj.findHeuristic()

destinationNode = None
q.put(QueueNode(source, None, 0, 0, None, totalBudget))
envObj.graph[source].g = 0

while(not q.empty()) :

	curr = q.get()
	if(curr.v != source) :
		print("curr = ", envObj.graph[curr.v].pos, ", parent = ", envObj.graph[curr.p.v].pos, ", g = ", curr.g, 
			", h = ", curr.h, ", (g+h) = ", (curr.g+curr.h))
		print("vehicle = ", curr.vehicle, ", budget = ", curr.budget)
	if(curr.v == goal) :
		destinationNode = curr
		break
	if(curr.budget <= 0) :
		continue

	envObj.graph[curr.v].visited = True;
	for adjNode in envObj.graph[curr.v].adj :
		if(not envObj.graph[adjNode.v].visited) :
			if(adjNode.w <= 3) :
				q.put(QueueNode(adjNode.v, curr, copy.copy(curr.g + (adjNode.w / cycleSpeed)), 
					envObj.graph[adjNode.v].h, "cycle", curr.budget))
			else :
				pathTime = curr.g + (adjNode.w / busSpeed)
				q.put(QueueNode(adjNode.v, curr, copy.copy(pathTime), envObj.graph[adjNode.v].h, "cycle", 
					curr.budget))
				if(curr.budget - (busFare * pathTime) > 0) :
					q.put(QueueNode(adjNode.v, curr, copy.copy(pathTime), envObj.graph[adjNode.v].h, "bus", 
						curr.budget - (busFare * pathTime)))
				

if(destinationNode is not None) :
	print ("goal reached")
	print ("Path : ")
	printPath(envObj, destinationNode)
	print()
else :
	print("Not reached")