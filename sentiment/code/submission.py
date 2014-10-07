#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *

############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction

def extractWordFeatures(x):
	"""
	Extract word features for a string x.
	@param string x:
	@return dict: feature vector representation of x.
	Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
	"""
	# BEGIN_YOUR_CODE (around 5 lines of code expected)
	return Counter(x.split())
	# END_YOUR_CODE

############################################################
# Problem 2b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor):
	'''
	Given |trainExamples| and |testExamples| (each one is a list of (x,y)
	pairs), a |featureExtractor| to apply to x, and the number of iterations to
	train |numIters|, return the weight vector (sparse feature vector) learned.

	You should implement stochastic gradient descent.

	Note: only use the trainExamples for training!
	You should call evaluatePredictor() on both trainExamples and testExamples
	to see how you're doing as you learn after each iteration.
	'''
	weights = {}  # feature => weight
	# BEGIN_YOUR_CODE (around 15 lines of code expected)
	def dhloss(w, i):
		x, y = trainExamples[i]
		phi = featureExtractor(x)
		margin = dotProduct(phi, w) * y
		grad = {}
		increment(grad, -y, phi)
		return grad if margin <=1 else {}
	def predictor(x):
		return dotProduct(featureExtractor(x), weights)
	def sgd(dF, n):
		numIters = 10
		eta = 1
		for it in range(numIters):
			for i in range(n):
				grad = dF(weights, i)
				increment(weights, -eta, grad)
	sgd(dhloss, len(trainExamples))
	# END_YOUR_CODE
	return weights

############################################################
# Problem 2c: generate test case

def generateDataset(numExamples, weights):
	'''
	Return a set of examples (phi(x), y) randomly which are classified correctly by
	|weights|.
	'''
	random.seed(42)
	# Return a single example (phi(x), y).
	# phi(x) can be anything (randomize!) with a nonzero score under the given weight vector
	# y should be 1 or -1 as classified by the weight vector.
	def generateExample():
			# BEGIN_YOUR_CODE (around 5 lines of code expected)
			numWeights = len(weights)
			weightKeys = weights.keys()
			phi = {random.choice(weightKeys) : 1 for i in range(5)}
			y = math.copysign(1, dotProduct(weights, phi))
			# END_YOUR_CODE
			return (phi, y)
	return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 2f: character features

def extractCharacterFeatures(n):
	'''
	Return a function that takes a string |x| and returns a sparse feature
	vector consisting of all n-grams of |x| without spaces.
	EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
	'''
	def extract(x):
			# BEGIN_YOUR_CODE (around 10 lines of code expected)
			x = x.replace(" ", "").replace("\t", "")
			return dict(Counter([x[i:i+n] for i in range(len(x) - n + 1)]))
			# END_YOUR_CODE
	return extract

############################################################
# Problem 2h: extra credit features

def extractExtraCreditFeatures(x):
	# BEGIN_YOUR_CODE (around 5 lines of code expected)
	raise Exception("Not implemented yet")
	# END_YOUR_CODE

############################################################
# Problem 3: k-means
############################################################


def kmeans(examples, K, maxIters):
	'''
	examples: list of examples, each example is a string-to-double dict representing a sparse vector.
	K: number of desired clusters
	maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
	Return: (length K list of cluster centroids,
					list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
					final reconstruction loss)
	'''
	# BEGIN_YOUR_CODE (around 35 lines of code expected)
	random.seed(42)
	clusters = [random.choice(examples) for i in range(K)]
	print clusters
	exLength = len(examples)
	kLength = len(clusters)
	# Create a list of zeros the same size as examples
	assignments = [0] * exLength
	def distance(phi, mu):
		v = dict.copy(phi)
		increment(v, -1, mu)
		return dotProduct(v, v)
	def loss():
		l = 0
		for e in range(exLength):
			k = assignments[e]
			l += distance(examples[e], clusters[k])
		return l
	def assignPointsToCentroids():
		for e in range(exLength):
			vals = [(distance(examples[e], clusters[k]), k) for k, val in enumerate(clusters)]
			print vals
			minVal, index = min(vals)
			print "Min " + str(minVal) + " at " + str(index)
			assignments[e] = index
	def bestCentroidForClusters():
		phiInCluster = 0
		vectorSum = {}
		for k in range(kLength):
			for e in range(exLength):
				if assignments[e] == k:
					phiInCluster += 1
					increment(vectorSum, 1, examples[e])
			print "VECTOR SUM"
			print vectorSum
			if phiInCluster == 0:
				phiInCluster = 1
			clusters[k] = {key : val/phiInCluster for key, val in vectorSum.items()}
			phiInCluster = 0
	for k in range(K):
		assignPointsToCentroids()
		bestCentroidForClusters()
	return (clusters, assignments, loss())
	# END_YOUR_CODE
