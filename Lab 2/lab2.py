import numpy as np 
import pickle

def loadRoad() :
	print("Loading road...")
	return pickle.load(open('road', 'r'))

def loadVehicle() :
	print("Loading vehicle...")
	return pickle.load(open('vehicle', 'r'))

def loadTime() :
	print("Loading time...")
	return pickle.load(open('time', 'r'))

class Environment :

	def __init__(self) :

	def calcSpeed(self, x) :
		return np.exp(0.5*x)/(1 + np.exp(0.5*x)) + 15/(1 + exp(0.5*x))
		