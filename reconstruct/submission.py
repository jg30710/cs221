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
		# Returns a triple (word, newState, cost)
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

	ucs = util.UniformCostSearch(verbose=0)
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
		# where state is (firstCompleteWord, secondIncomplete, index)
		return (wordsegUtil.SENTENCE_BEGIN, self.queryWords[0], 0)
		# END_YOUR_CODE

	def isGoal(self, state):
		# BEGIN_YOUR_CODE (around 5 lines of code expected)
		index = state[2]
		return index >= len(self.queryWords)
		# END_YOUR_CODE

	def succAndCost(self, state):
		# BEGIN_YOUR_CODE (around 10 lines of code expected)
		# choices are triples (words, newState, cost)
		choices = []
		word, afterWord, index = state
		nextIndex = index + 1
		possibles = self.possibleFills(afterWord)
		# Handle case for a word with no vowel substitutions
		possibles.add(afterWord)
		nextWord = ''
		if nextIndex < len(self.queryWords):
			nextWord = self.queryWords[nextIndex]
		for poss in possibles:
			choices.append((poss, (poss, nextWord, nextIndex), self.bigramCost(word, poss)))
		return choices
		# END_YOUR_CODE

def insertVowels(queryWords, bigramCost, possibleFills):
	# BEGIN_YOUR_CODE (around 5 lines of code expected)
	if len(queryWords) == 0:
		return ''
	ucs = util.UniformCostSearch(verbose=0)
	ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))
	if ucs.actions is not None:
		return ' '.join(ucs.actions)
	return ''
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
		return (wordsegUtil.SENTENCE_BEGIN, 0)
		# END_YOUR_CODE

	def isGoal(self, state):
		# BEGIN_YOUR_CODE (around 5 lines of code expected)
		index = state[1]
		return index >= len(self.query)
		# END_YOUR_CODE

	def succAndCost(self, state):
		# BEGIN_YOUR_CODE (around 15 lines of code expected)
		choices = []
		q = self.query
		prevWord, currIndex = state
		for index in range(currIndex + 1, len(q) + 1):
			word = q[currIndex:index]
			possibles = self.possibleFills(word)
			for poss in possibles:
				choices.append((poss, (poss, index), self.bigramCost(prevWord, poss)))
		return choices
		# END_YOUR_CODE

def segmentAndInsert(query, bigramCost, possibleFills):
	if len(query) == 0:
		return ''

	# BEGIN_YOUR_CODE (around 5 lines of code expected)
	ucs = util.UniformCostSearch(verbose=0)
	ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills))
	return ' '.join(ucs.actions)
	# END_YOUR_CODE

############################################################

if __name__ == '__main__':
	shell.main()
