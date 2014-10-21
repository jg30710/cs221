from submission import *
from util import *
from collections import Counter

 #4b stuff
 #Policy from value iteration
vi = ValueIteration()
vi.solve(largeMDP)
viPolicy = vi.pi

ql = QLearningAlgorithm(largeMDP.actions, largeMDP.discount(), identityFeatureExtractor, 0)
rewards = simulate(largeMDP, ql, 30000, 1000, False, False)
ql.explorationProb = 0

# Compare
totalInPolicy = len(viPolicy)
qActions = Counter()
actions = Counter()
actionCounter = Counter()
differentActions = 0
for state, action in viPolicy.items():
	qAction = ql.getAction(state)
	qActions[qAction] += 1
	actions[action] += 1
	if action != qAction:
		differentActions += 1
print "Percent different " + str( (differentActions * 1.0)/totalInPolicy )

print actions, qActions

# 4d

#vi = ValueIteration()
#vi.solve(originalMDP)
#viPolicy = vi.pi

#rl = FixedRLAlgorithm(viPolicy)
#ql = QLearningAlgorithm(originalMDP.actions, originalMDP.discount(), identityFeatureExtractor, 0)
#rRewards = simulate(newThresholdMDP, rl, 30000, 1000, False, False)
#qRewards = simulate(newThresholdMDP, ql, 30000, 1000, False, False)

#def avgReward(r):
	#return sum([i for i in r])/len(r)

#print "Relaxed " + str(avgReward(rRewards))
#print "Q " + str(avgReward(qRewards))
