# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


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


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        foodList = newFood.asList()
        ghostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
        ghostDistances = [manhattanDistance(newPos, ghost) for ghost in ghostPositions]
        scaredGhosts = [newScaredTimes[i] > 0 for i in range(len(newGhostStates))]
        
        score = successorGameState.getScore()
        
        # Food factor: prioritize getting closer to food
        # Food factor: prioritize getting closer to food
        if foodList:
            minFoodDist = min([manhattanDistance(newPos, food) for food in foodList])
            if minFoodDist == 0:
                score += 100  # Reward eating food immediately
            else:
                score += 15 / minFoodDist  # Higher weight for food proximity
        
        # Ghost avoidance: penalize being close to non-scared ghosts
        for i, dist in enumerate(ghostDistances):
            if not scaredGhosts[i]:
                if dist < 2:
                    score -= 1000  # Severe penalty for danger zone
                else:
                    score -= 25 / dist  # Dynamic penalty based on proximity
        
        # Scared ghost chasing: incentivize eating scared ghosts
        for i, dist in enumerate(ghostDistances):
            if scaredGhosts[i]:
                score += 50 / (1 + dist)  # Strong incentive for close scared ghosts
        
        # Avoid stopping unless absolutely necessary
        if action == Directions.STOP:
            score -= 10
        
        return score

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

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        def minimax(state, depth, agentIndex):
            # If game over or max depth reached, return evaluation function value
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            # Pacman's turn (maximize)
            if agentIndex == 0:
                return maxValue(state, depth)
            # Ghosts' turn (minimize)
            else:
                return minValue(state, depth, agentIndex)
        
        def maxValue(state, depth):
            bestScore = float('-inf')
            bestAction = None
            
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                score = minimax(successor, depth, 1)  # Next agent is ghost 1
                
                if score > bestScore:
                    bestScore = score
                    bestAction = action
            
            if depth == 0:  # Only return the action at the root
                return bestAction
            return bestScore
        
        def minValue(state, depth, agentIndex):
            bestScore = float('inf')
            numAgents = state.getNumAgents()
            
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                
                # If last ghost, go back to Pacman at next depth level
                if agentIndex == numAgents - 1:
                    score = minimax(successor, depth + 1, 0)
                else:
                    score = minimax(successor, depth, agentIndex + 1)
                
                bestScore = min(bestScore, score)
            return bestScore
        
        return minimax(gameState, 0, 0)


        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphabeta(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            if agentIndex == 0:
                return maxValue(state, depth, alpha, beta)
            else:
                return minValue(state, depth, agentIndex, alpha, beta)
        
        def maxValue(state, depth, alpha, beta):
            bestScore = float('-inf')
            bestAction = None
            
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                score = alphabeta(successor, depth, 1, alpha, beta)
                
                if score > bestScore:
                    bestScore = score
                    bestAction = action
                
                alpha = max(alpha, bestScore)
                if bestScore > beta:
                    break  # Prune branch
            
            return bestAction if depth == 0 else bestScore
        
        def minValue(state, depth, agentIndex, alpha, beta):
            bestScore = float('inf')
            numAgents = state.getNumAgents()
            
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                
                if agentIndex == numAgents - 1:
                    score = alphabeta(successor, depth + 1, 0, alpha, beta)
                else:
                    score = alphabeta(successor, depth, agentIndex + 1, alpha, beta)
                
                bestScore = min(bestScore, score)
                beta = min(beta, bestScore)
                if bestScore < alpha:
                    break  # Prune branch
            
            return bestScore
        
        return alphabeta(gameState, 0, 0, float('-inf'), float('inf'))

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            if agentIndex == 0:
                return maxValue(state, depth)
            else:
                return expValue(state, depth, agentIndex)
        
        def maxValue(state, depth):
            bestScore = float('-inf')
            bestAction = None
            
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                score = expectimax(successor, depth, 1)
                
                if score > bestScore:
                    bestScore = score
                    bestAction = action
            
            return bestAction if depth == 0 else bestScore
        
        def expValue(state, depth, agentIndex):
            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)
            
            numAgents = state.getNumAgents()
            probability = 1.0 / len(actions)
            expectedScore = 0
            
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                
                if agentIndex == numAgents - 1:
                    score = expectimax(successor, depth + 1, 0)
                else:
                    score = expectimax(successor, depth, agentIndex + 1)
                
                expectedScore += probability * score
            
            return expectedScore
        
        return expectimax(gameState, 0, 0)

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
        # Extract relevant game state information
    pacmanPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghost.scaredTimer for ghost in ghostStates]
    
    score = currentGameState.getScore()
    foodList = foodGrid.asList()
    
    # Food Distance: Encourage eating food
    if foodList:
        minFoodDist = min(manhattanDistance(pacmanPos, food) for food in foodList)
        score += 10 / (1 + minFoodDist)  # Higher weight for closer food
    
    # Capsule Distance: Incentivize eating power pellets
    if capsules:
        minCapsuleDist = min(manhattanDistance(pacmanPos, cap) for cap in capsules)
        score += 15 / (1 + minCapsuleDist)
    
    # Ghost Avoidance: Penalize close ghosts unless they are scared
    for i, ghost in enumerate(ghostStates):
        ghostDist = manhattanDistance(pacmanPos, ghost.getPosition())
        if scaredTimes[i] > 0:
            score += 20 / (1 + ghostDist)  # Encourage eating scared ghosts
        elif ghostDist < 2:
            score -= 1000  # Heavy penalty if too close to an active ghost
        else:
            score -= 10 / ghostDist  # Mild penalty for proximity to active ghosts
    
    return score


    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

