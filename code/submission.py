import collections

############################################################
# Problem 3a

def computeMaxWordLength(text):
	"""
		Given a string |text|, return the longest word in |text|.  If there are
		ties, choose the word that comes latest in the alphabet. There won't be 
		puctuations and there will only be splits on spaces. You might find
		max() and list comprehensions handy here.
	"""
	# BEGIN_YOUR_CODE (around 1 line of code expected)
	return max(sorted(text.split(), None, None, reverse=True), key=lambda x: len(x))
	# END_YOUR_CODE

############################################################
# Problem 3b

def createExistsFunction(text):
	"""
		Given a text, return a function f, where f(word) returns whether |word|
		occurs in |text| or not.  f should run in O(1) time.  You might find it
		useful to use set().
	"""
	# BEGIN_YOUR_CODE (around 4 lines of code expected)
	textSet = set(text.split())
	return (lambda word : word in textSet)
	# END_YOUR_CODE

############################################################
# Problem 3c

def manhattanDistance(loc1, loc2):
	"""
		Return the Manhattan distance between two locations, where locations are
		pairs (e.g., (3, 5)).
	"""
	# BEGIN_YOUR_CODE (around 1 line of code expected)
	return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])
	# END_YOUR_CODE

############################################################
# Problem 3d

def sparseVectorDotProduct(v1, v2):
	"""
	Given two sparse vectors |v1| and |v2|, each represented as Counters, return
	their dot product.
	You might find it useful to use sum() and a list comprehension.
	"""
	# BEGIN_YOUR_CODE (around 4 lines of code expected)
	components = set(v1.keys()) | set(v2.keys())
	return sum([v1[component] * v2[component] for component in components])
	# END_YOUR_CODE

############################################################
# Problem 3e

def incrementSparseVector(v1, scale, v2):
	"""
		Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
	"""
	# BEGIN_YOUR_CODE (around 2 lines of code expected)
	for component in v2: v2[component] *= scale
	return v1.update(v2)
	# END_YOUR_CODE

############################################################
# Problem 3f

def computeMostFrequentWord(text):
	"""
		Splits the string |text| by whitespace and returns two things as a pair: 
				the set of words that occur the maximum number of times, and
		their count, i.e.
		(set of words that occur the most number of times, that maximum number/count)
		You might find it useful to use collections.Counter().
	"""
	# BEGIN_YOUR_CODE (around 5 lines of code expected)
	textCounter = collections.Counter(text.split(" "))
	mostCommonTuple = textCounter.most_common(1)[0]
	count = mostCommonTuple[1]
	ties = set([tie[0] for tie in textCounter.most_common() if tie[1] >= count])
	return (ties, len(ties))
	# END_YOUR_CODE

############################################################
# Problem 3g

def computeLongestPalindrome(text):
	"""
		A palindrome is a string that is equal to its reverse (e.g., 'ana').
		Compute the length of the longest palindrome that can be obtained by deleting
		letters from |text|.
		For example: the longest palindrome in 'animal' is 'ama'.
		Your algorithm should run in O(len(text)^2) time.
		You should first define a recurrence before you start coding.
	"""
	# BEGIN_YOUR_CODE (around 19 lines of code expected)
	cache = {}
	def isPalindrome(t):
		return t == t[::-1];
	def removeCharAtIndex(t, i):
		return t[:i] + t[i+1:]
	def recurse(t,m,n):
		if t in cache:
			return cache[t]
		if isPalindrome(t):
			ans = len(t)
		elif t[m] == t[n]:
			ans = recurse(t, m + 1, n - 1)
		else:
			ans = max(recurse(removeCharAtIndex(t, n), m, n-1), recurse(removeCharAtIndex(t, m), m, n-1))
		cache[t] = ans
		return ans
	return recurse(text,0, len(text) -1)
	# END_YOUR_CODE
