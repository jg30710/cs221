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
	def dloss(w, i):
		x, y = trainExamples[i]
		phi = featureExtractor(x)
		grad = {}
		increment(grad, 2 * (dotProduct(phi, w) - y), phi)
		return grad
	def predictor(x):
		return dotProduct(featureExtractor(x), weights)
	def sgd(dF, n):
		numIters = 10
		eta = 0.1
		for it in range(numIters):
			for i in range(n):
				grad = dF(weights, i)
			increment(weights, -eta, grad)
	# END_YOUR_CODE
	sgd(dloss, len(trainExamples))
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
			raise Exception("Not implemented yet")
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
			raise Exception("Not implemented yet")
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
	raise Exception("Not implemented yet")
	# END_YOUR_CODE
