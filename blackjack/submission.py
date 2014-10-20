import collections, util, math, random

############################################################

############################################################
# Problem 2a

class ValueIteration(util.MDPAlgorithm):

	# Implement value iteration.  First, compute V_opt using the methods 
	# discussed in class.  Once you have computed V_opt, compute the optimal 
	# policy pi.  Note that ValueIteration is an instance of util.MDPAlgrotithm, 
	# which means you will need to set pi and V (see util.py).
	def solve(self, mdp, epsilon=0.001):
		mdp.computeStates()
		# BEGIN_YOUR_CODE (around 15 lines of code expected)
		self.pi = {}
		self.V = {state : 0 for state in mdp.states}
		VPrevious = {state: float('inf') for state, val in self.V.items()}
		def valueDifference():
			return max(abs(self.V[s] - VPrevious[s]) for s, v in self.V.items())
		while(valueDifference() > epsilon):
			VPrevious = dict(self.V)
			for s in mdp.states:
				Qs = []
				for a in mdp.actions(s):
					Q = 0
					for sPrime in mdp.succAndProbReward(s, a):
						newState, prob, reward = sPrime
						Q += prob * (reward + mdp.discount() * self.V[newState])
					Qs.append( (Q, a) )
				self.V[s], self.pi[s] = max(Qs)
		# END_YOUR_CODE
        

############################################################
# Problem 2b

# If you decide 2b is true, prove it in writeup.pdf and put "return None" for
# the code blocks below.  If you decide that 2b is false, construct a
# counterexample by filling out this class and returning an alpha value in
# counterexampleAlpha().
class CounterexampleMDP(util.MDP):
	def __init__(self, n):
		# BEGIN_YOUR_CODE (around 5 lines of code expected)
		# Defaulting this value to 5
		self.n = 5
		# END_YOUR_CODE

	def startState(self):
		# BEGIN_YOUR_CODE (around 5 lines of code expected)
		return 0
		# END_YOUR_CODE

	# Return set of actions possible from |state|.
	def actions(self, state):
		# BEGIN_YOUR_CODE (around 5 lines of code expected)
		return [-1, 1]
		# END_YOUR_CODE

	# Return a list of (newState, prob, reward) tuples corresponding to edges
	# coming out of |state|.
	def succAndProbReward(self, state, action):
		# BEGIN_YOUR_CODE (around 5 lines of code expected)
		T = [0.4, 0.6]
		return [(state, T[1], 0),
				(min(max(state + action, -self.n), +self.n), T[0], state)]
		# END_YOUR_CODE

	def discount(self):
		# BEGIN_YOUR_CODE (around 5 lines of code expected)
		return 0.9
		# END_YOUR_CODE

def counterexampleAlpha():
	# BEGIN_YOUR_CODE (around 5 lines of code expected)
	return 1
	# END_YOUR_CODE

# Test stuff for counter example
#alg = ValueIteration()
## Norm
#alg.solve(CounterexampleMDP(2, False), .0001)
#print alg.V
#alg = ValueIteration()
#alg.solve(CounterexampleMDP(2, True), .0001)
#print alg.V

############################################################
# Problem 3a

class BlackjackMDP(util.MDP):
	def __init__(self, cardValues, multiplicity, threshold, peekCost):
		"""
		cardValues: array of card values for each card type
		multiplicity: number of each card type
		threshold: maximum total before going bust
		peekCost: how much it costs to peek at the next card
		"""
		self.cardValues = cardValues
		self.multiplicity = multiplicity
		self.threshold = threshold
		self.peekCost = peekCost

	# Return the start state.
	# Look at this function to learn about the state representation.
	# The first element of the tuple is the sum of the cards in the player's
	# hand.  The second element is the next card, if the player peeked in the
	# last action.  If they didn't peek, this will be None.  The final element
	# is the current deck.
	def startState(self):
		return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

	# Return set of actions possible from |state|.
	def actions(self, state):
		return ['Take', 'Peek', 'Quit']

	# Return a list of (newState, prob, reward) tuples corresponding to edges
	# coming out of |state|.  Indicate a terminal state (after quitting or
	# busting) by setting the deck to (0, None).
	def succAndProbReward(self, state, action):
		# BEGIN_YOUR_CODE (around 55 lines of code expected)
		edges = []
		total, nextCardIndex, multiplicityTuple = state
		if multiplicityTuple is None:
			return edges
		# Quit 
		if action == 'Quit':
			edges.append( ((total, None, None), 1, total)  )
			return edges
		multiplicitySize = len(multiplicityTuple)
		cardsLeft = sum([counts for counts in multiplicityTuple])
		if cardsLeft == 0:
			return edges
		# Handle peek
		if action == "Peek":
			# Handle peeking twice
			if nextCardIndex is not None:
				return edges
			for index in range(multiplicitySize):
				if multiplicityTuple[index] != 0:
					prob = multiplicityTuple[index]/(cardsLeft * 1.0)
					edges.append( ((total, index, multiplicityTuple), prob, -1 * self.peekCost) )
		# Take
		if action == "Take":
			if nextCardIndex is not None:
				newTotal = total + self.cardValues[nextCardIndex]
				tupleCopy = ()
				tupleCopy = list(multiplicityTuple)
				tupleCopy[nextCardIndex] -= 1
				tupleCopy = tuple(tupleCopy)
				edges.append( ((newTotal, None, tupleCopy), 1, 0) )
				return edges
			for index in range(multiplicitySize):
				prob = multiplicityTuple[index]/(cardsLeft * 1.0)
				if multiplicityTuple[index] != 0:
					tupleCopy = ()
					tupleCopy = list(multiplicityTuple)
					tupleCopy[index] -= 1
					tupleCopy = tuple(tupleCopy)
					newTotal = total + self.cardValues[index]
					if newTotal > self.threshold:
						# Bust
						edges.append( ((newTotal, None, None), prob, 0) )
					else:
						if cardsLeft == 1:
							# If there's nothing left after this draw
							edges.append( ((newTotal, None, None), 1, newTotal) )
						else:
							edges.append( ((newTotal, None, tupleCopy), prob, 0) )
		return edges
		# END_YOUR_CODE

	def discount(self):
		return 1

############################################################
# Problem 3b

def peekingMDP():
	"""
	Return an instance of BlackjackMDP where peeking is the optimal action at
	least 10% of the time.
	"""
	# BEGIN_YOUR_CODE (around 5 lines of code expected)
	mdp = BlackjackMDP(cardValues=[3, 4, 17], multiplicity=4,
                                   threshold=20, peekCost=1)
	return mdp
	# END_YOUR_CODE

############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
	def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
		self.actions = actions
		self.discount = discount
		self.featureExtractor = featureExtractor
		self.explorationProb = explorationProb
		self.weights = collections.Counter()
		self.numIters = 0

	# Return the Q function associated with the weights and features
	def getQ(self, state, action):
		score = 0
		for f, v in self.featureExtractor(state, action):
			score += self.weights[f] * v
		return score

	# This algorithm will produce an action given a state.
	# Here we use the epsilon-greedy algorithm: with probability
	# |explorationProb|, take a random action.
	def getAction(self, state):
		self.numIters += 1
		if random.random() < self.explorationProb:
			return random.choice(self.actions(state))
		else:
			return max((self.getQ(state, action), action) for action in self.actions(state))[1]

	# Call this function to get the step size to update the weights.
	def getStepSize(self):
		return 1.0 / math.sqrt(self.numIters)

	# We will call this function with (s, a, r, s'), which you should use to update |weights|.
	# Note that if s is a terminal state, then s' will be None.  Remember to check for this.
	# You should update the weights using self.getStepSize(); use
	# self.getQ() to compute the current estimate of the parameters.
	def incorporateFeedback(self, state, action, reward, newState):
		# BEGIN_YOUR_CODE (around 15 lines of code expected)
		residual = 0
		if newState is None:
			residual = reward - self.getQ(state, action)
		else:
			residual = reward + \
				self.discount * max([self.getQ(newState, newAction) for newAction in self.actions(state)]) - \
				self.getQ(state, action)
		features = self.featureExtractor(state, action) # list of (feature name, feature value)
		for key, val in features:
			self.weights[key] = self.weights[key]	+ self.getStepSize() * residual * val
		# END_YOUR_CODE

# Return a singleton list containing indicator feature for the (state, action)
# pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
	featureKey = (state, action)
	featureValue = 1
	return [(featureKey, featureValue)]

############################################################
# Problem 4b: convergence of Q-learning

# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)

############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs (see
# identityFeatureExtractor()).
# Implement the following features:
# - indicator on the total and the action (1 feature).
# - indicator on the presence/absence of each card and the action (1 feature).  Only add this feature if the deck != None
# - indicator on the number of cards for each card type and the action (len(counts) features).  Only add these features if the deck != None
def blackjackFeatureExtractor(state, action):
	total, nextCard, counts = state
	# BEGIN_YOUR_CODE (around 10 lines of code expected)
	extractor = []
	extractor.append(((total, action), 1))
	if counts is not None:
		presenceKey = (tuple(1 if c > 0 else 0 for c in counts), action)
		extractor.append( (presenceKey, 1) )
		for index, count in enumerate(counts):
			# number of card per type and action
			extractor.append( ((index, count, action), 1) )
	return extractor
	# END_YOUR_CODE

############################################################
# Problem 4d: changing mdp

# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)
