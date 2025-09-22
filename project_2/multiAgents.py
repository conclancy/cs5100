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

        # Return successorGameState.getScore()
        return score
        

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
            """Helper function to cycle through agents"""
            return (agentIndex + 1) % numAgents

        def nextDepth(agentIndex, depth):
            """Helper function to increment depth when the next agent is Pac-Man (Index = 0)"""
            return depth + 1 if nextAgent(agentIndex) == 0 else depth

        def isTerminal(state, agentIndex, depth):
            """Function to stop expansion if we reach a win or lose state or the tree is fully exapanded"""
            return state.isWin() or state.isLose() or (depth == self.depth and agentIndex == 0)

        def minimax(state, agentIndex, depth):
            """Recursive minimax function which returns value and best action"""
            # Test if starting in a terminal state 
            if isTerminal(state, agentIndex, depth):
                return self.evaluationFunction(state), None

            # If no legal moves exists set state to terminal
            legal = state.getLegalActions(agentIndex)
            if not legal:
                return self.evaluationFunction(state), None

            # Maximize Pac-Man's value
            if agentIndex == 0:
                bestVal, bestAct = -float('inf'), None
                for a in legal:
                    succ = state.generateSuccessor(agentIndex, a)
                    val, _ = minimax(succ, nextAgent(agentIndex), nextDepth(agentIndex, depth))
                    if val > bestVal:
                        bestVal, bestAct = val, a
                return bestVal, bestAct
            # Minimize the value for the ghosts
            else:  
                bestVal, bestAct = float('inf'), None
                for a in legal:
                    succ = state.generateSuccessor(agentIndex, a)
                    val, _ = minimax(succ, nextAgent(agentIndex), nextDepth(agentIndex, depth))
                    if val < bestVal:
                        bestVal, bestAct = val, a
                return bestVal, bestAct

        # Recurse and start the search at the root
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
            """Helper function to cycle through agents"""
            return (agentIndex + 1) % numAgents

        def nextDepth(agentIndex, depth):
            """Helper function to increment depth when the next agent is Pac-Man (Index = 0)"""
            return depth + 1 if nextAgent(agentIndex) == 0 else depth

        def isTerminal(state, agentIndex, depth):
            """Function to stop expansion if we reach a win or lose state or the tree is fully exapanded"""
            return state.isWin() or state.isLose() or (depth == self.depth and agentIndex == 0)

        def alphabeta(state, agentIndex, depth, alpha, beta):
            """Recursive alpha-beta function which returns value and best action"""
            # Test if starting in a terminal state 
            if isTerminal(state, agentIndex, depth):
                return self.evaluationFunction(state), None

            # If no legal moves exists set state to terminal
            legal = state.getLegalActions(agentIndex)
            if not legal:
                return self.evaluationFunction(state), None

            # Maximize Pac-Man's alpha
            if agentIndex == 0:
                value, bestAct = -9999, None
                for a in legal:
                    succ = state.generateSuccessor(agentIndex, a)
                    childVal, _ = alphabeta(succ, nextAgent(agentIndex), nextDepth(agentIndex, depth), alpha, beta)
                    # Update when childVal improves current value
                    if childVal > value:
                        value, bestAct = childVal, a
                    # Prune to avoid tie
                    if value > beta:
                        return value, bestAct  
                    # Tighten alpha based on max value
                    alpha = max(alpha, value)
                return value, bestAct
            # Minimize ghost beta
            else:
                value, bestAct = 9999, None
                for a in legal:
                    succ = state.generateSuccessor(agentIndex, a)
                    childVal, _ = alphabeta(succ, nextAgent(agentIndex), nextDepth(agentIndex, depth), alpha, beta)
                    # Update when childValue minimize current value 
                    if childVal < value:
                        value, bestAct = childVal, a
                    # Prune to avoid tie
                    if value < alpha:
                        return value, bestAct
                    # Tighten beta based on min value
                    beta = min(beta, value)
                return value, bestAct

        # Recurse and start the search at the root
        _, action = alphabeta(gameState, 0, 0, -9999, 9999)
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
            """Helper function to cycle through agents"""
            return (agentIndex + 1) % numAgents

        def nextDepth(agentIndex, depth):
            """Helper function to increment depth when the next agent is Pac-Man (Index = 0)"""
            return depth + 1 if nextAgent(agentIndex) == 0 else depth

        def isTerminal(state, agentIndex, depth):
            """Function to stop expansion if we reach a win or lose state or the tree is fully exapanded"""
            return state.isWin() or state.isLose() or (depth == self.depth and agentIndex == 0)

        def expectimax(state, agentIndex, depth):
            """Recursive expectimax which returns value and best action"""
            # Test if starting in a terminal state 
            if isTerminal(state, agentIndex, depth):
                return self.evaluationFunction(state), None

            # If no legal moves exists set state to terminal
            legal = state.getLegalActions(agentIndex)
            if not legal:
                return self.evaluationFunction(state), None

            # Maximize Pac-Man's value 
            if agentIndex == 0:
                bestVal, bestAct = -9999, None
                for a in legal:
                    succ = state.generateSuccessor(agentIndex, a)
                    val, _ = expectimax(succ, nextAgent(agentIndex), nextDepth(agentIndex, depth))
                    if val > bestVal:
                        bestVal, bestAct = val, a
                return bestVal, bestAct
            # Minimize Ghost's values
            else:
                total = 0.0
                for a in legal:
                    succ = state.generateSuccessor(agentIndex, a)
                    val, _ = expectimax(succ, nextAgent(agentIndex), nextDepth(agentIndex, depth))
                    total += val
                expVal = total / float(len(legal))
                return expVal, None

        # Recurse and start the search at the root
        _, action = expectimax(gameState, 0, 0)
        return action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Heurisitic for the current state that valances fast food consumption,
    safe ghost avoidance, and opportunistic capsult/scared-ghost chasing

    Features calculated on current state:
    - Use inverse-distance to create smooth gradients and avoids extreme values
    - Base Score: start from currentGameState.getScore()
    - Food Progress: +2.5/(d_min_food + 1) to pull Pac-Man towards the nearest dot, and 
        -0.5 * food to reward clearing th board. Wanted higher than capsules with pressure
        to make progress on the board.
    - Capsules: +1.5/(d_min_capsules + 1) to move towards power pellets and -1 * Capsules
        to encourage using them rather than orbiting. Less weight than food so Pac-Man does
        not detour too far to reach a capsule
    - Gosts: 
        -- If scared: (3+.02 * scaredTimer)/(distance + 1) to chase edible ghosts
        -- If active: strong penalty for ghost at short range, with weaker penalty further out
    """
    # Determine if the current state is terminal
    if currentGameState.isWin():
        return 9999
    if currentGameState.isLose():
        return -9999

    # set state variables for use in function
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghosts = currentGameState.getGhostStates()

    # Use build in game score to respect true wards 
    score = currentGameState.getScore()

    # Food distance - pull toward nearest dot 
    if food:
        # Inverse scaling to give smooth curve 
        minFood = min(manhattanDistance(pos, f) for f in food)
        # Use weight of 2.5 to pull Pac-Man towards nearest dot
        score += 2.5 / (minFood + 1)
        # Fewer dots left is better with presure to finish the board
        score -= 0.5 * len(food)
    else:
        # Strong weight for clearing all dots from board
        score += 10

    # Capsules- encourage move safely towards nearby opportunities
    if capsules:
        minCap = min(manhattanDistance(pos, c) for c in capsules)
        # Set weight less than food to prevent detouring 
        score += 1.5 / (minCap + 1)
        # Weight consuming over orbittng capsules 
        score -= 1.0 * len(capsules)

    # Ghost interactions for both scared and active states 
    for g in ghosts:
        gpos = g.getPosition()
        dist = manhattanDistance(pos, gpos)
        # Scared ghost logic
        if g.scaredTimer > 0:
            # Approach edible ghosts; weight by remaining scared time
            score += (3.0 + 0.02 * g.scaredTimer) / (dist + 1)
        # Active ghost logic
        else:
            # Strong penalty for being within 1-2 tiles. Use large score so that no 
            # combination of positive terms creates a suicide situation for Pac-Man
            if dist <= 1:
                score -= 100
            elif dist == 2:
                score -= 10
            # Mild general repulsion for active ghosts that are further away
            score -= 1.0 / (dist + 1)

    # Return the final score with higher values being better for Pac-Man
    return score

# Abbreviation
better = betterEvaluationFunction

