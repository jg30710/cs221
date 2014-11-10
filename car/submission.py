'''
Licensing Information: Please do not distribute or publish solutions to this
project. You are free to use and extend Driverless Car for educational
purposes. The Driverless Car project was developed at Stanford, primarily by
Chris Piech (piech@cs.stanford.edu). It was inspired by the Pacman projects.
'''
from engine.const import Const
import util, math, random, collections

# Class: ExactInference
# ---------------------
# Maintain and update a belief distribution over the probability of a car
# being in a tile using exact updates (correct, but slow times).
class ExactInference(object):
    
    # Function: Init
    # --------------
    # Constructer that initializes an ExactInference object which has
    # numRows x numCols number of tiles.
    def __init__(self, numRows, numCols):
        self.skipElapse = False ### ONLY USED BY GRADER.PY in case problem 3 has not been completed
        self.belief = util.Belief(numRows, numCols)
        self.transProb = util.loadTransProb()
   
    # Function: Observe (reweight the probablities based on an observation)
    # -----------------
    # Takes |self.belief| and updates it based on the distance observation
    # $d_t$ and your position $a_t$.
    #
    # - agentX: x location of your car (not the one you are tracking)
    # - agentY: y location of your car (not the one you are tracking)
    # - observedDist: true distance plus a mean-zero Gaussian with standard deviation Const.SONAR_STD
    # 
    # Notes:
    # - Convert row and col indices into locations using util.rowToY and util.colToX.
    # - util.pdf: computes the probability density function for a Gaussian
    # - util.Belief is a class that represents the belief for a single
    #   inference state of a single car (see util.py).
    # - Don't forget to normalize self.belief!
    def observe(self, agentX, agentY, observedDist):
        # BEGIN_YOUR_CODE (around 10 lines of code expected)
        def euclidean(x1, y1, x2, y2):
            return math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        # Compute the pdf off of observed Dist
        # For every tile in the grid
        for r in range(self.belief.getNumRows()):
            for c in range(self.belief.getNumCols()):
                otherX, otherY = (util.colToX(c), util.rowToY(r))
                dist = euclidean(agentX, agentY, otherX, otherY)
                prev = self.belief.getProb(r, c)
                pdf = util.pdf(observedDist, Const.SONAR_STD, dist)
                # Update the posterior probability
                posterior = pdf * prev
                self.belief.setProb(r, c, posterior)
        self.belief.normalize()
        # END_YOUR_CODE

    # Function: Elapse Time (propose a new belief distribution based on a learned transition model)
    # ---------------------
    # Takes |self.belief| and updates it based on the passing of one time step.
    # Notes:
    # - Use the transition probabilities in self.transProb.
    # - Use self.belief.addProb and self.belief.getProb to manipulate beliefs.
    # - Don't forget to normalize self.belief!
    def elapseTime(self):
        if self.skipElapse: return ### ONLY FOR THE GRADER TO USE IN Problem 2
        # BEGIN_YOUR_CODE (around 10 lines of code expected)
        beliefCopy = util.Belief(self.belief.getNumRows(), self.belief.getNumCols(), 0.0)
        for key, transProb in self.transProb.items():
            if transProb != 0:
                oldTile, newTile = key
                oldR, oldC = oldTile
                r, c = newTile
                prevPosterior = self.belief.getProb(oldR, oldC)
                posterior = prevPosterior * transProb
                beliefCopy.addProb(r, c, posterior)
        beliefCopy.normalize()
        self.belief = beliefCopy
        # END_YOUR_CODE
      
    # Function: Get Belief
    # ---------------------
    # Returns your belief of the probability that the car is in each tile. Your
    # belief probabilities should sum to 1.    
    def getBelief(self):
        return self.belief

        
# Class: Particle Filter
# ----------------------
# Maintain and update a belief distribution over the probability of a car
# being in a tile using a set of particles.
class ParticleFilter(object):
    
    NUM_PARTICLES = 200
    
    # Function: Init
    # --------------
    # Constructer that initializes an ParticleFilter object which has
    # numRows x numCols number of tiles.
    def __init__(self, numRows, numCols):
        self.belief = util.Belief(numRows, numCols)

        # Load the transition probabilities and store them in a dict of Counters
        # self.transProbDict[oldTile][newTile] = probability of transitioning from oldTile to newTile
        self.transProb = util.loadTransProb()
        self.transProbDict = dict()
        for (oldTile, newTile) in self.transProb:
            if not oldTile in self.transProbDict:
                self.transProbDict[oldTile] = collections.Counter()
            self.transProbDict[oldTile][newTile] = self.transProb[(oldTile, newTile)]
            
        # Initialize the particles randomly
        self.particles = collections.Counter()
        potentialParticles = self.transProbDict.keys()
        for i in range(self.NUM_PARTICLES):
            particleIndex = int(random.random() * len(potentialParticles))
            self.particles[potentialParticles[particleIndex]] += 1
            
        self.updateBelief()

    # Function: Update Belief
    # ---------------------
    # Updates |self.belief| with the probability that the car is in each tile
    # based on |self.particles|, which is a Counter from particle to
    # probability (which should sum to 1).
    def updateBelief(self):
        newBelief = util.Belief(self.belief.getNumRows(), self.belief.getNumCols(), 0)
        for tile in self.particles:
            newBelief.setProb(tile[0], tile[1], self.particles[tile])
        newBelief.normalize()
        self.belief = newBelief

    # Function: Observe:
    # -----------------
    # Takes |self.particles| and updates them based on the distance observation
    # $d_t$ and your position $a_t$.  You should:
    # - Reweight the particles based on the observation.
    # - Resample the particles.
    #
    # - agentX: x location of your car (not the one you are tracking)
    # - agentY: y location of your car (not the one you are tracking)
    # - observedDist: true distance plus a mean-zero Gaussian with standard deviation Const.SONAR_STD
    #
    # Notes:
    # - Create |self.NUM_PARTICLES| new particles during resampling.
    # - To pass the grader, you must call util.weightedRandomChoice() once per new particle.
    def observe(self, agentX, agentY, observedDist):
        # BEGIN_YOUR_CODE (around 15 lines of code expected)
        def euclidean(x1, y1, x2, y2):
            return math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        weights = {}
        posteriors = {}
        for tile, occurrences in self.particles.items():
            weights[tile] = 0
            r, c = tile 
            pX, pY = (util.colToX(c), util.rowToY(r))
            dist = euclidean(agentX, agentY, pX, pY)
            pdf = util.pdf(observedDist, Const.SONAR_STD, dist)
            posteriors[tile] = pdf * self.belief.getProb(r, c)
            weights[tile] = pdf * occurrences
        newParticles = collections.Counter()
        for p in range(self.NUM_PARTICLES):
            tile = util.weightedRandomChoice(weights)
            newParticles[tile] += 1
        self.particles = newParticles
        for tile in self.particles:
            self.belief.setProb(r, c, posteriors[tile])
        # END_YOUR_CODE
        self.updateBelief()

    # Function: Elapse Time (propose a new belief distribution based on a learned transition model)
    # ---------------------
    # Read |self.particles| (Counter) corresonding to time $t$ and writes
    # |self.particles| corresponding to time $t+1$.
    #
    # Notes:
    # - transition probabilities |self.transProbDict|
    # - Use util.weightedRandomChoice() to sample a new particle.
    # - To pass the grader, you must loop over the particles using
    #       for tile in self.particles
    #   and call util.weightedRandomChoice() once per particle on the tile.
    def elapseTime(self):
        # BEGIN_YOUR_CODE (around 10 lines of code expected)
        particlesCopy = collections.Counter(self.particles)
        weights = {}
        for tile in self.particles:
            transitionTiles = util.weightedRandomChoice(self.transProbDict)
            transProb = self.transProbDict[transitionTiles]
            oldTile, newTile = transitionTiles
            weights[newTile] = transProb
        newParticles = collections.Counter()
        for tile in self.particles:
            newTile = util.weightedRandomChoice(weights)
            newParticles[newTile] += 1
        print newParticles
        self.particles = newParticles
        # END_YOUR_CODE
        
    # Function: Get Belief
    # ---------------------
    # Returns your belief of the probability that the car is in each tile. Your
    # belief probabilities should sum to 1.    
    def getBelief(self):
        return self.belief
