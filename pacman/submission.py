from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
	"""
		A reflex agent chooses an action at each choice point by examining
		its alternatives via a state evaluation function.

		The code below is provided as a guide.  You are welcome to change
		it in any way you see fit, so long as you don't touch our method
		headers.
	"""
	def __init__(self):
		self.lastPositions = []
		self.dc = None


	def getAction(self, gameState):
		"""
		getAction chooses among the best options according to the evaluation function.

		getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
		------------------------------------------------------------------------------
		Description of GameState and helper functions:

		A GameState specifies the full game state, including the food, capsules,
		agent configurations and score changes. In this function, the |gameState| argument 
		is an object of GameState class. Following are a few of the helper methods that you 
		can use to query a GameState object to gather information about the present state 
		of Pac-Man, the ghosts and the maze.

		gameState.getLegalActions(): 
				Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

		gameState.generateSuccessor(agentIndex, action): 
				Returns the successor state after the specified agent takes the action. 
				Pac-Man is always agent 0.

		gameState.getPacmanState():
				Returns an AgentState object for pacman (in game.py)
				state.pos gives the current position
				state.direction gives the travel vector

		gameState.getGhostStates():
				Returns list of AgentState objects for the ghosts

		gameState.getNumAgents():
				Returns the total number of agents in the game


		The GameState class is defined in pacman.py and you might want to look into that for 
		other helper methods, though you don't need to.
		"""
		# Collect legal moves and successor states
		legalMoves = gameState.getLegalActions()

		# Choose one of the best actions
		scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
		bestScore = max(scores)
		bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
		chosenIndex = random.choice(bestIndices) # Pick randomly among the best




		return legalMoves[chosenIndex]

	def evaluationFunction(self, currentGameState, action):
		"""
		The evaluation function takes in the current and proposed successor
		GameStates (pacman.py) and returns a number, where higher numbers are better.

		The code below extracts some useful information from the state, like the
		remaining food (oldFood) and Pacman position after moving (newPos).
		newScaredTimes holds the number of moves that each ghost will remain
		scared because of Pacman having eaten a power pellet.
		"""
		# Useful information you can extract from a GameState (pacman.py)
		successorGameState = currentGameState.generatePacmanSuccessor(action)
		newPos = successorGameState.getPacmanPosition()
		oldFood = currentGameState.getFood()
		newGhostStates = successorGameState.getGhostStates()
		newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


		return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
	"""
		This default evaluation function just returns the score of the state.
		The score is the same one displayed in the Pacman GUI.

		This evaluation function is meant for use with adversarial search agents
		(not reflex agents).
	"""
	return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
	"""
		This class provides some common elements to all of your
		multi-agent searchers.  Any methods defined here will be available
		to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

		You *do not* need to make any changes here, but you can if you want to
		add functionality to all your adversarial search agents.  Please do not
		remove anything, however.

		Note: this is an abstract class: one that should not be instantiated.  It's
		only partially specified, and designed to be extended.  Agent (game.py)
		is another abstract class.
	"""

	def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
		self.index = 0 # Pacman is always agent index 0
		self.evaluationFunction = util.lookup(evalFn, globals())
		self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
	"""
		Your minimax agent (problem 1)
	"""

	def getAction(self, gameState):
		"""
			Returns the minimax action from the current gameState using self.depth
			and self.evaluationFunction. Terminal states can be found by one of the following: 
			pacman won, pacman lost or there are no legal moves. 

			Here are some method calls that might be useful when implementing minimax.

			gameState.getLegalActions(agentIndex):
				Returns a list of legal actions for an agent
				agentIndex=0 means Pacman, ghosts are >= 1

			Directions.STOP:
				The stop direction, which is always legal

			gameState.generateSuccessor(agentIndex, action):
				Returns the successor game state after an agent takes an action

			gameState.getNumAgents():
				Returns the total number of agents in the game

			gameState.isWin():
				Returns True if it's a winning state

			gameState.isLose():
				Returns True if it's a losing state

			self.depth:
				The depth to which search should continue
		"""

		# BEGIN_YOUR_CODE (around 30 lines of code expected)
		numAgents = gameState.getNumAgents()
		def recurse(gameState, agent, depth):
			# Base case
			if gameState.isWin() or gameState.isLose():
				# Return the utility
				return (gameState.getScore(), None)
			# choices is a list of (score, action) tuples
			legalActions = gameState.getLegalActions(agent)
			choices = []
			if depth == 0:
				# Max depth reached, call eval function
				choices = [(self.evaluationFunction(gameState.generateSuccessor(agent, action)), \
						action) for action in legalActions if action != Directions.STOP]
			else:
				# Only advance the depth once ALL agents have gone through
				nextDepth = depth
				# Probably could have been clever with modulus, but I'd rather
				# be explicit and clear than clever
				if agent != numAgents -1:
					nextAgent = agent + 1
				else:
					nextAgent = 0
					nextDepth -= 1
				choices = [(recurse(gameState \
						.generateSuccessor(agent, action), nextAgent, nextDepth)[0], \
						action) for action in legalActions if action != Directions.STOP]
			if agent == self.index: # We're pacman
				return max(choices)
			else:
				return min(choices)
		return recurse(gameState, self.index, self.depth)[1]
		# END_YOUR_CODE

######################################################################################
# Problem 2a: implementing alpha-beta
class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
		Your minimax agent with alpha-beta pruning (problem 2)
	"""

	def getAction(self, gameState):
		"""
			Returns the minimax action using self.depth and self.evaluationFunction
		"""

		# BEGIN_YOUR_CODE (around 50 lines of code expected)
		numAgents = gameState.getNumAgents()
		alpha = float("-inf")
		beta = float("inf")
		def overlap(a, b):
			return b > a
		def recurse(gameState, agent, depth, alpha, beta):
			# Base case
			if gameState.isWin() or gameState.isLose():
				# Return the utility
				return (gameState.getScore(), None)
			# choices is a list of (score, action) tuples
			legalActions = gameState.getLegalActions(agent)
			choices = []
			if depth == 0:
				# Max depth reached, call eval function
				choices = [(self.evaluationFunction(gameState.generateSuccessor(agent, action)), \
						action) for action in legalActions if action != Directions.STOP]
			else:
				# Only advance the depth once ALL agents have gone through
				# Probably could have been clever with modulus, but I'd rather
				# be explicit and clear than clever
				if agent != numAgents -1:
					nextAgent = agent + 1
				else:
					nextAgent = 0
					depth -= 1
				for action in legalActions:
					if action != Directions.STOP:
						score = recurse(gameState.generateSuccessor(agent, action), \
								nextAgent, depth, alpha, beta)[0]
						if agent == self.index:
							if score > alpha:
								alpha = score
							choices.append( (score, action) )
						else:
							if score < beta:
								# New lower bound
								beta = score
							if not overlap(alpha, beta):
								# No trivial overlap
								return (beta, action)
							choices.append( (score, action) )
			if agent == self.index: # We're pacman
				return max(choices)
			else:
				return min(choices)
		return recurse(gameState, self.index, self.depth, alpha, beta)[1]
		# END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
	"""
		Your expectimax agent (problem 3)
	"""

	def getAction(self, gameState):
		"""
			Returns the expectimax action using self.depth and self.evaluationFunction

			All ghosts should be modeled as choosing uniformly at random from their
			legal moves.
		"""

		# BEGIN_YOUR_CODE (around 25 lines of code expected)
		numAgents = gameState.getNumAgents()
		def recurse(gameState, agent, depth):
			# Base case
			if gameState.isWin() or gameState.isLose():
				# Return the utility
				return (gameState.getScore(), None)
			# choices is a list of (score, action) tuples
			legalActions = gameState.getLegalActions(agent)
			numActions = len(legalActions)
			choices = []
			randomActionChoiceProb = 1/(numActions * 1.0) if numActions != 0 else 1
			ghostScoreSum = 0
			if depth == 0:
				# Max depth reached, call eval function
				for action in legalActions:
					if action != Directions.STOP:
						score = self.evaluationFunction(gameState\
								.generateSuccessor(agent, action))
						if agent != self.index:
							score *= randomActionChoiceProb
							ghostScoreSum += score
						choices.append( (score, action) )
			else:
				# Only decrement the depth once ALL agents have gone through
				nextDepth = depth
				if agent != numAgents -1:
					nextAgent = agent + 1
				else:
					nextAgent = 0
					nextDepth -= 1
				for action in legalActions:
					if action != Directions.STOP:
						score = recurse(gameState \
								.generateSuccessor(agent, action), nextAgent, nextDepth)[0]
						if agent != self.index:
							score *= randomActionChoiceProb
							ghostScoreSum += score
						else:
							choices.append( (score, action) )
			if agent == self.index: # We're pacman
				random.shuffle(choices)
				return max(choices)
			else:
				return (ghostScoreSum, random.choice(legalActions))
		
		mini = recurse(gameState, self.index, self.depth)
		print mini
		return mini[1]
		# END_YOUR_CODE

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
	"""
		Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
		evaluation function (problem 4).

		The metrics I use to improve Pacman's performance are computing distance to
		the nearest food (and inverting that quantity) as well as computing the
		most food in a particular direction. I had hoped that it would cause pacman
		to jitter less (further I wish I had access to pacman's previous moves overall)
		but it seemed to have carried on regardless. The heuristic created helps pacman
		win most of the time (again both being algorithms that produce quantities related
		to distance to food) and is hopefully performant enough for extra credit!
	"""

	# BEGIN_YOUR_CODE (around 30 lines of code expected)
	# Useful information you can extract from a GameState (pacman.py)
	"""
	"""
	walls = currentGameState.getWalls()
	width = walls.width
	height = walls.height
	dirToIndex = {"North": 1, "South": 1, "East": 0, "West": 0}
	dirToIncrement = {"North": 1, "South": -1, "East": 1, "West": -1}
	dirToInverse = {"North": "South", "South": "North", "East": "West", "West": "East"}
	def distanceToFood(pos):
		# Compute the distance to the closest food
		distance = float('inf')
		foodLeft = False
		for x in range(width):
			for y in range(height):
				if currentGameState.hasFood(x, y):
					foodLeft = True
					md = manhattanDistance((x,y), pos)
					if md < distance:
						distance = md
		distance = distance if foodLeft else 0.01
		return distance
	
	def follow(pos, direction):
		posCopy = list(pos)
		dirIndex = dirToIndex[direction]
		foodCount = 0
		posCopy[dirIndex] += dirToIncrement[direction]
		while not currentGameState.hasWall(posCopy[0], posCopy[1]) and\
				currentGameState.hasFood(posCopy[0], posCopy[1]) and\
				( (dirIndex == 0 and posCopy[dirIndex] < width and posCopy[dirIndex] > 0) or\
						(dirIndex == 1 and posCopy[dirIndex] < height and posCopy[dirIndex] > 0) ):
			foodCount += 1
			posCopy[dirIndex] += dirToIncrement[direction]
		return foodCount

	def foodInDirection(pos, direction):
		posCopy = list(pos)
		dirIndex = dirToIndex[direction]
		posCopy[dirIndex] += dirToIncrement[direction]
		return currentGameState.hasFood(posCopy[0], posCopy[1])

	def foodBehind(pos, direction):
		posCopy = list(pos)
		behind = dirToInverse[direction]
		dirIndex = dirToIndex[behind]
		posCopy[dirIndex] += dirToIncrement[behind]
		return currentGameState.hasFood(posCopy[0], posCopy[1])

	def getFoodInDirection(pos, direction):
		# Find the direction with the best food prospects
		start = [pos[0], pos[1]]
		end = [0, 0]
		if direction is "North":
			start = [0, pos[1]]
			end = [width, height]
		elif direction is "South":
			start = [0, pos[1]]
			end = [width, 0]
		elif direction is "East":
			start = [pos[0], 0]
			end = [width, height]
		elif direction is "West":
			start = [pos[0], 0]
			end = [0, height]
		else:
			start = [0, 0]
			end = [width, height]
		count = 0
		totalFood = currentGameState.getNumFood()
		for x in range(start[0], end[0]):
			for y in range(start[1], end[1]):
				if currentGameState.hasFood(x, y):
					count += 1
		totalFood = totalFood if totalFood != 0 else -1
		# Returns the percentage of food in a given direction
		return count/(totalFood * 1.0)

	# Current pacman state
	pacmanState = currentGameState.getPacmanState()
	pacmanCurrPos = pacmanState.getPosition()
	pacmanCurrDir = pacmanState.getDirection()
	score = currentGameState.getScore()
	#foodDirScore = getFoodInDirection(pacmanCurrPos, pacmanCurrDir)
	#foodDirScore *= 0.3 * abs(score)
	#foodDirScore = 0
	#fa = foodAround(pacmanCurrPos, pacmanCurrDir)
	#fb = foodBehind(pacmanCurrPos, pacmanCurrDir)
	dtf = distanceToFood(pacmanCurrPos)
	# measure distance to food as a fraction of the overall score.
	# remember that a losing move is a move just as well and needs
	# to be regarded highly.
	#penalty = 1
	#if fb:
		#penalty = 1/100
	fd = follow(pacmanCurrPos, pacmanCurrDir)
	if fd == 0:
		fd = 0.1
	if dtf == 0:
		dtf = 0.1
	dtf *= 1.0
	dtf = ((1/dtf)**2) * 0.6 * abs(score)
	total = dtf + score - 1/fd

	return total
	# END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction

