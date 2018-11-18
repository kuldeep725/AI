import math, random, string
import numpy as np

target = "kuldeeplovesgeneticalgorithm"

def diff(s1, s2) :
	sum = 0
	for i in range(len(s1)) :
		sum += (ord(s1[i])-ord(s2[i]))**2
	return math.sqrt(sum)

def getRandomString() :
	# s = ''.join(random.SystemRandom().choice(string.ascii_lowercase + string.ascii_uppercase + string.digits) for _ in range(N))
	# return s.replace('[0-9]', ' ')
	s = ""
	for i in range(len(target)) :
		if(target[i] != ' ') :
			random_char = np.random.randint(ord('a'), ord('z'))
			s += chr(random_char)
		# else :
		# 	s += ' '
	return s

# selection
def selection(offsprings) :
	best = None
	best2 = None
	min_val = 1e8
	min_val2 = 1e8
	for offspring in offsprings : 
		val = diff(offspring, target) 
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
	rand_index1 = np.random.randint(len(s))
	rand_index2 = np.random.randint(len(s))
	sList = list(s)
	# sList[rand_index1], sList[rand_index2] = sList[rand_index2], sList[rand_index1]
	sList[rand_index1] = chr(np.random.randint(ord('a'), ord('z')))
	return ''.join(sList)

def generatePopulation(n, k) :
	populations = [] 
	for i in range(n) :
		population.append(getRandomString(k))
	return populations

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

parent1 = getRandomString()
parent2 = getRandomString()

print("parent1 = ", parent1)
print("parent2 = ", parent2)
iter = 0
while(iter < 1000) :
	if((parent1 == target) or (parent2 == target)) :
		print("Final parent1, Final parent2 = ", parent1, parent2)
		print("Reached in iter", iter)
		break
	print("parent1, parent2 = ", parent1, parent2)
	parent1, parent2 = v(parent1, parent2)
	iter += 1

print("end")