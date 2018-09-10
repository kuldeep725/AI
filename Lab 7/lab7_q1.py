# Member : (1) Kuldeep Singh Bhandari
#		   (2) Amit Vikram Singh
from queue import PriorityQueue 
class QueueNode :
	def __init__(self, v, p) :
		self.v = v 		# vertex
		self.p = p		# parent
		self.g = g		# total cost till now
		self.h = h		# cost to go

	def __cmp__(self, other) :
		return cmp(self.g + self.h, other.g + other.h)

class Node :
	def __init__(self, v, w) :
		self.v = v 		# vertex
		self.w = w 		# weight

class GraphNode :
	def __init__ (self, adj=[], g = INF, h = 0, visited = false) :
		self.adj = adj
		self.g = g
		self.h = h
		self.visited = visited

	def __cmp__(self, other) :
		return cmp(self.g + self.h, other.g + other.h)

class Environment :

	def __init__(self, source, goal, n, m):
		self.source = source
		self.goal = goal
		graph = [ GraphNode([]) for i in range(n+1)]

n, m = map(int, input().split())
q = PriorityQueue()
source = 0
goal = 5
envObj = Environment(source, goal, n, m)
adj = [ [] for _ in range(n+1)]
for i in range(0, m) :
	u, v, w = map(int, input().split())
	envObj.graph[u].adj.append(Node(v, w))
	envObj.graph[v].adj.append(Node(u, w))
	# adj[u].append(Node(v, w))
	# adj[v].append(Node(u, w))


destinationNode = None
exploredList = [False for _ in range(n+1)]
q.put(QueueNode(source, -1, 0, 0))

while(not q.empty()) :
	curr = q.get()
	if(curr.v == goal) :
		destinationNode = curr
		break

	curr.visited = True
	for adjNode in adj[curr.v] :
		if(not exploredList[adjNode.v]) :
			exploredList[adjNode.v] = True