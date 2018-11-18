# ### Lab Quiz 2
# ####  Problem :  Bowling First
# ##### Name	:  Kuldeep Singh Bhandari(111601009
## Logic :  We are doing Bowling Strategy using Value Iteration. We have a state of 7-tuple :
		#  Total Overs Left,  Wickets Left, Overs left for bowler 1, 2, 3, 4, 5
		# We are checking in each state, which bowler should be chosen - that is our policy 
		# and we are storing runs as our value of the state. The aim is to find the most optimal policy for
		# each state
		# Using dynamic programming bottom-top approach, we will finally get the optimal policy for each state
from __future__ import print_function
import numpy as np
import itertools



overLeftList = [0, 1, 2]
TOT_OVERS = 11
TOT_WICKETS = 4
bowlers_over_left = 0

class Environment :

	def __init__(self, P_wickets, Runs) :
		self.P_wickets = P_wickets
		self.Runs = Runs
		self.policy = np.zeros((11, 4, 3, 3, 3, 3, 3), dtype = np.int8)
		self.V = np.zeros((11, 4, 3, 3, 3, 3, 3))

	def ValueIteration(self) :
		for over_left in range(1, TOT_OVERS) :
			for wicket_left in range(1, TOT_WICKETS) :
				for (b1, b2, b3, b4, b5) in itertools.product(overLeftList, overLeftList, overLeftList, overLeftList, overLeftList) :
					minRun = 1e7
					bestBowler = -1
					currState = (over_left, wicket_left, b1, b2, b3, b4, b5)
					actionList = [b1, b2, b3, b4, b5]
					for bowler in [0, 1, 2, 3, 4] :
						bowlers_over_left = actionList[bowler]
						if(bowlers_over_left == 0) :
							continue
						run = self.Runs[bowler]
						wicketState = [over_left-1, wicket_left-1, b1, b2, b3, b4, b5]
						wicketState[2+bowler] -= 1
						notWicketState = [over_left-1, wicket_left, b1, b2, b3, b4, b5]
						notWicketState[2 + bowler] -= 1
						run += (self.P_wickets[bowler] * self.V[tuple(wicketState)]) + ((1-self.P_wickets[bowler]) * self.V[tuple(notWicketState)])
						if(minRun > run) :
							minRun = run
							bestBowler = bowler

					self.policy[currState] = bestBowler
					if(minRun == 1e7) :
						self.V[currState] = 0
					else :
						self.V[currState] = minRun

P_wickets = np.array([6.0/33, 6.0/30, 6.0/24, 6.0/18, 6.0/15])		
Runs = np.array([3, 3.5, 4, 4.5, 5])
envObj = Environment(P_wickets, Runs)
envObj.ValueIteration()

for over_left in range(1, 11) :
	for wicketLeft in range(1, 4) :
		for (b1, b2, b3, b4, b5) in itertools.product (overLeftList, overLeftList, overLeftList, overLeftList, overLeftList) :
			currState = (over_left, wicketLeft, b1, b2, b3, b4, b5)
			print((over_left, wicketLeft, b1, b2, b3, b4, b5), envObj.policy[currState], envObj.V[currState])