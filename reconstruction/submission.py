import shell
import util
import wordsegUtil

############################################################
# Problem 1b: Solve the segmentation problem under a unigram model

class SegmentationProblem(util.SearchProblem):
	def __init__(self, query, unigramCost):
		self.query = query
		self.unigramCost = unigramCost

	def startState(self):
		# BEGIN_YOUR_CODE (around 5 lines of code expected)
		# Start at the 0th index
		return 0
		# END_YOUR_CODE

	def isGoal(self, state):
		# BEGIN_YOUR_CODE (around 5 lines of code expected)
		return state == len(self.query)
		# END_YOUR_CODE

	def succAndCost(self, state):
		# Returns a tripe (word, newState, cost)
		# BEGIN_YOUR_CODE (around 10 lines of code expected)
		choices = []
		q = self.query
		for index in range(1, len(q) + 1):
			word = q[state:index]
			choices.append((word, index, self.unigramCost(word)))
		return choices
		# END_YOUR_CODE

def segmentWords(query, unigramCost):
	if len(query) == 0:
		return ''

	ucs = util.UniformCostSearch(verbose=3)
	ucs.solve(SegmentationProblem(query, unigramCost))

	# BEGIN_YOUR_CODE (around 5 lines of code expected)
	return ' '.join(ucs.actions)
	# END_YOUR_CODE

############################################################
# Problem 2b: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
	def __init__(self, queryWords, bigramCost, possibleFills):
		self.queryWords = queryWords
		self.bigramCost = bigramCost
		self.possibleFills = possibleFills

	def startState(self):
		# BEGIN_YOUR_CODE (around 5 lines of code expected)
		raise Exception("Not implemented yet")
		# END_YOUR_CODE

	def isGoal(self, state):
		# BEGIN_YOUR_CODE (around 5 lines of code expected)
		raise Exception("Not implemented yet")
		# END_YOUR_CODE

	def succAndCost(self, state):
		# BEGIN_YOUR_CODE (around 10 lines of code expected)
		raise Exception("Not implemented yet")
		# END_YOUR_CODE

def insertVowels(queryWords, bigramCost, possibleFills):
	# BEGIN_YOUR_CODE (around 5 lines of code expected)
	raise Exception("Not implemented yet")
	# END_YOUR_CODE

############################################################
# Problem 3b: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
	def __init__(self, query, bigramCost, possibleFills):
		self.query = query
		self.bigramCost = bigramCost
		self.possibleFills = possibleFills

	def startState(self):
		# BEGIN_YOUR_CODE (around 5 lines of code expected)
		raise Exception("Not implemented yet")
		# END_YOUR_CODE

	def isGoal(self, state):
		# BEGIN_YOUR_CODE (around 5 lines of code expected)
		raise Exception("Not implemented yet")
		# END_YOUR_CODE

	def succAndCost(self, state):
		# BEGIN_YOUR_CODE (around 15 lines of code expected)
		raise Exception("Not implemented yet")
		# END_YOUR_CODE

def segmentAndInsert(query, bigramCost, possibleFills):
	if len(query) == 0:
		return ''

	# BEGIN_YOUR_CODE (around 5 lines of code expected)
	raise Exception("Not implemented yet")
	# END_YOUR_CODE

############################################################

if __name__ == '__main__':
	shell.main()
