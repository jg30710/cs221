'''
	Ex 1
	Reduce to subproblem
	Insert into s <=> deletion from t
	Order right to left
		delete(s) => a ca , the cats
		delete(t) => a cat , the cat
		sub => a ca, the cat
	Use indices rather than strings!
'''
def editDistance(s, t):
	# recurse(m, n): the edit distance between first m
	# letters of s and

	cache = {}
	def recurse(m, n):
		if (m, n) in cache:
			return cache[(m,n)]
		if m == 0:
			ans = n
		elif n == 0:
			ans = m
		elif s[m-1] == t[n-1]:
			ans = recurse(m-1, n-1)
		else:
			ans = 1 + min(recurse(m-1, n), recurse(m, n-1),
					recurse(m-1, n-1))
			cache[(m,n)] = ans
		return ans
	recurse(len(s), len(t))

print editDistance("a cat!", "the cats!")

'''
	Ex 2
	Abstract away from details
	Gradient descent! Computing the derivative to find the direction
	of the solution.
	w <- w - r*f'
'''

def gradientDescent(F, dF):
		w = 0
		numIters = 100
		eta = 0.01
		for t in range(numIters):
				value = F(w)
				gradient = dF(w)
				w = w - eta * gradient
		return w

points = [(2,4), (4,2)]
def F(w):
		#return (w - 5)**2
		return sum((x*w - y)**2 for x, y in points)

def dF(w):
		#return 2 * (w - 5)
		return sum(2 * (x*w - y) * x for x, y in points)

print gradientDescent(F, dF)

