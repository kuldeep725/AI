import math, random, string
import numpy as np

# logic :
#    Initialize two numbers in between -10 and 10. Then we run cross over, mutation and selection with fitness given by function 'diff'
	# finally value converges to optimal value

def diff(s) :
	sum = 0
	first = int(s, 2)**2
	second = (int(s, 2)-2)**2
	return math.sqrt(first + second)

def getRandomString(N) :
	# s = ''.join(random.SystemRandom().choice(string.ascii_lowercase + string.ascii_uppercase + string.digits) for _ in range(N))
	# return s.replace('[0-9]', ' ')
	s = ""
	for i in range(N) :
		s += np.random.randint(2)
	return s

# selection
def selection(offsprings) :
	best = None
	best2 = None
	min_val = 1e8
	min_val2 = 1e8
	for offspring in offsprings : 
		val = diff(offspring)
		if(val < min_val) :
			min_val = val
			best = offspring
		elif(val < min_val2) :
			min_val2 = val
			best2 = offspring

	if(best2 == None) : best2 = best
	return best, best2

# single point crossover
def crossover(s1, s2) :
	l1 = np.random.randint(len(s1))
	l2 = np.random.randint(len(s1))
	offspring1 = s1[:l1] + s2[l1:]
	offspring2 = s2[:l2] + s1[l2:]
	return offspring1, offspring2

# mutation
def mutation(s) :
	rand_index = np.random.randint(len(s))
	sList = list(s)
	# sList[rand_index1], sList[rand_index2] = sList[rand_index2], sList[rand_index1]
	if(sList[rand_index] == '1'):
		sList[rand_index] = '0'
	else :
		sList[rand_index] = '1'
	return ''.join(sList)

def v(s1, s2) :
	if(np.random.rand() >= 0.5) :
		s1, s2 = crossover(s1, s2)

	populations = []
	for i in range(len(s1)) :
		if(np.random.rand() >= 0.5) :
			populations.append(mutation(s1))
		else :
			populations.append(s1)
		if(np.random.rand() >= 0.5) :
			populations.append(mutation(s2))
		else :
			populations.append(s2)

	return selection(populations)

first, second = np.random.randint(0, 10, 2)

parent1 = bin(first).split('b')[1]
parent2 = bin(second).split('b')[1]

n1 = 4-len(parent1)
n2 = 4-len(parent2)
while(n1) :
	parent1 += '0'
	n1-=1

while(n2) :
	parent2 += '0'
	n2-=1

print("initial parent1 = ", parent1)
print("initial parent2 = ", parent2)
iter = 0
best = None
while(iter < 1000) :
	
	print("parent1, parent2 = ", parent1, parent2)
	parent1, parent2 = v(parent1, parent2)
	if(diff(parent1) < diff(parent2)) :
		best = parent1
	else :
		best = parent2
	iter += 1

print("best value ", int(best, 2))
print("end")