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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        capsules = successorGameState.getCapsules()

        score = successorGameState.getScore()

        # Penalize stopping
        if action == Directions.STOP:
            score -= 2


        # Food feature: inverse of distance to closest food
        foodList = newFood.asList()
        if foodList:
            closestFood = min(manhattanDistance(newPos, f) for f in foodList)
            score += 1.5 / (closestFood + 1)

        # Capsule feature: incentive to get closer
        if capsules:
            closestCap = min(manhattanDistance(newPos, c) for c in capsules)
            score += 1.0 / (closestCap + 1)


        # Ghost features
        for g in newGhostStates:
            gpos = g.getPosition()
            dist = manhattanDistance(newPos, gpos)
            if g.scaredTimer > 0:
                # Attract when edible (but avoid division by zero)
                score += 2.0 / (dist + 1)
            else:
                # Strongly repel dangerous ghosts when very close
                if dist <= 1:
                    score -= 10
                score -= 0.5 / (dist + 1)

        return score
        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def nextAgent(agentIndex):
            return (agentIndex + 1) % numAgents

        def nextDepth(agentIndex, depth):
            return depth + 1 if nextAgent(agentIndex) == 0 else depth

        def isTerminal(state, agentIndex, depth):
            return state.isWin() or state.isLose() or (depth == self.depth and agentIndex == 0)

        def minimax(state, agentIndex, depth):
            if isTerminal(state, agentIndex, depth):
                return self.evaluationFunction(state), None

            legal = state.getLegalActions(agentIndex)
            if not legal:
                return self.evaluationFunction(state), None

            if agentIndex == 0:  # Maximize for Pacman
                bestVal, bestAct = -float('inf'), None
                for a in legal:
                    succ = state.generateSuccessor(agentIndex, a)
                    val, _ = minimax(succ, nextAgent(agentIndex), nextDepth(agentIndex, depth))
                    if val > bestVal:
                        bestVal, bestAct = val, a
                return bestVal, bestAct
            else:  # Minimize for ghosts
                bestVal, bestAct = float('inf'), None
                for a in legal:
                    succ = state.generateSuccessor(agentIndex, a)
                    val, _ = minimax(succ, nextAgent(agentIndex), nextDepth(agentIndex, depth))
                    if val < bestVal:
                        bestVal, bestAct = val, a
                return bestVal, bestAct

        _, action = minimax(gameState, 0, 0)
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        numAgents = gameState.getNumAgents()

        def nextAgent(agentIndex):
            return (agentIndex + 1) % numAgents

        def nextDepth(agentIndex, depth):
            return depth + 1 if nextAgent(agentIndex) == 0 else depth

        def isTerminal(state, agentIndex, depth):
            return state.isWin() or state.isLose() or (depth == self.depth and agentIndex == 0)

        def alphabeta(state, agentIndex, depth, alpha, beta):
            if isTerminal(state, agentIndex, depth):
                return self.evaluationFunction(state), None

            legal = state.getLegalActions(agentIndex)
            if not legal:
                return self.evaluationFunction(state), None

            if agentIndex == 0:  # Max node
                value, bestAct = -float('inf'), None
                for a in legal:
                    succ = state.generateSuccessor(agentIndex, a)
                    childVal, _ = alphabeta(succ, nextAgent(agentIndex), nextDepth(agentIndex, depth), alpha, beta)
                    if childVal > value:
                        value, bestAct = childVal, a
                    if value > beta:
                        return value, bestAct  # prune on strict > to avoid tie-pruning
                    alpha = max(alpha, value)
                return value, bestAct
            else:  # Min node (ghost)
                value, bestAct = float('inf'), None
                for a in legal:
                    succ = state.generateSuccessor(agentIndex, a)
                    childVal, _ = alphabeta(succ, nextAgent(agentIndex), nextDepth(agentIndex, depth), alpha, beta)
                    if childVal < value:
                        value, bestAct = childVal, a
                    if value < alpha:
                        return value, bestAct  # prune on strict < to avoid tie-pruning
                    beta = min(beta, value)
                return value, bestAct

        _, action = alphabeta(gameState, 0, 0, -float('inf'), float('inf'))
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        numAgents = gameState.getNumAgents()

        def nextAgent(agentIndex):
            return (agentIndex + 1) % numAgents

        def nextDepth(agentIndex, depth):
            return depth + 1 if nextAgent(agentIndex) == 0 else depth

        def isTerminal(state, agentIndex, depth):
            return state.isWin() or state.isLose() or (depth == self.depth and agentIndex == 0)

        def expectimax(state, agentIndex, depth):
            if isTerminal(state, agentIndex, depth):
                return self.evaluationFunction(state), None

            legal = state.getLegalActions(agentIndex)
            if not legal:
                return self.evaluationFunction(state), None

            if agentIndex == 0:  # Max node
                bestVal, bestAct = -float('inf'), None
                for a in legal:
                    succ = state.generateSuccessor(agentIndex, a)
                    val, _ = expectimax(succ, nextAgent(agentIndex), nextDepth(agentIndex, depth))
                    if val > bestVal:
                        bestVal, bestAct = val, a
                return bestVal, bestAct
            else:  # Chance node (uniform)
                total = 0.0
                for a in legal:
                    succ = state.generateSuccessor(agentIndex, a)
                    val, _ = expectimax(succ, nextAgent(agentIndex), nextDepth(agentIndex, depth))
                    total += val
                expVal = total / float(len(legal))
                return expVal, None

        _, action = expectimax(gameState, 0, 0)
        return action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return -float('inf')

    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghosts = currentGameState.getGhostStates()

    score = currentGameState.getScore()

    # Food distance (closer is better)
    if food:
        minFood = min(manhattanDistance(pos, f) for f in food)
        score += 2.5 / (minFood + 1)
        # Fewer food left is better
        score -= 0.5 * len(food)
    else:
        score += 10  # reward clearing all food

    # Capsules: encourage picking them up / moving closer
    if capsules:
        minCap = min(manhattanDistance(pos, c) for c in capsules)
        score += 1.5 / (minCap + 1)
        score -= 1.0 * len(capsules)

    # Ghost interactions
    for g in ghosts:
        gpos = g.getPosition()
        dist = manhattanDistance(pos, gpos)
        if g.scaredTimer > 0:
            # Approach edible ghosts; weight by remaining scared time
            score += (3.0 + 0.02 * g.scaredTimer) / (dist + 1)
        else:
            # Strong penalty for being within 1-2 tiles
            if dist <= 1:
                score -= 100
            elif dist == 2:
                score -= 10
            # Mild general repulsion
            score -= 1.0 / (dist + 1)

    return score

# Abbreviation
better = betterEvaluationFunction
