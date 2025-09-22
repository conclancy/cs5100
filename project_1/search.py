# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    """Graph-search DFS using util.Stack. Returns a list of actions."""
    # Test if the start state is the goal
    start = problem.getStartState()
    if problem.isGoalState(start):
        return []

    # Initialize the fringe for the start node
    fringe = util.Stack()
    fringe.push((start, [], set()))

    # Initialize set to hold graph search info
    visited = set()

    # DFS loop, pop the nearest node and test for goal
    while not fringe.isEmpty():
        state, path, pathVisited = fringe.pop()

        # Skip if thi state has already been visited
        if state in visited:
            continue
        visited.add(state)

        # Test to see if the current state is the goal
        if problem.isGoalState(state):
            return path

        # Expand each unvisted successor
        for succ, action, stepCost in problem.getSuccessors(state):
            if succ not in visited:
                fringe.push((succ, path + [action], pathVisited | {state}))

    # If goal is not found, the return an empty list
    return []

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    # Test if the start state is the goal
    start = problem.getStartState()
    if problem.isGoalState(start):
        return []

    # Initialize a queue and add the starting state 
    fringe = util.Queue()
    fringe.push((start, []))

    # Initialize set to hold graph search info
    visited = set([start])

    # BFS loop, pop from queue to test for goal
    while not fringe.isEmpty():
        state, path = fringe.pop()

        # Test duqueued state to see if it is the goal
        if problem.isGoalState(state):
            return path

        # Expand each unvisted successor 
        for succ, action, stepCost in problem.getSuccessors(state):
            if succ not in visited:
                visited.add(succ)
                fringe.push((succ, path + [action]))

    # If goal is not found, the return an empty list
    return []

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    # Test if the start state is the goal
    start = problem.getStartState()
    if problem.isGoalState(start):
        return []

    # Initiate a priority queue ordered by "g"
    fringe = util.PriorityQueue()
    fringe.push((start, [], 0), 0)

    # Initiate a variable to hold lowest cost for eachs tate
    best_g = {start: 0}

    # UCS loop, pop from queue to test for goal
    while not fringe.isEmpty():
        state, path, g = fringe.pop()

        # Skip if we’ve already found a cheaper way to state
        if g > best_g.get(state, float('inf')):
            continue
        
        # Test current sate for goal 
        if problem.isGoalState(state):
            return path

        # Relax edges and try to improve the costs of "g's" neighbors 
        for succ, action, stepCost in problem.getSuccessors(state):
            new_g = g + stepCost
            if new_g < best_g.get(succ, float('inf')):
                best_g[succ] = new_g
                fringe.push((succ, path + [action], new_g), new_g)

    # If goal is not found, the return an empty list
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # Test if the start state is the goal
    start = problem.getStartState()
    if problem.isGoalState(start):
        return []

    # Initiate a priority queue ordered by "f"
    fringe = util.PriorityQueue()
    start_h = heuristic(start, problem)
    fringe.push((start, [], 0), start_h)

    # Initiate a variable to hold lowest cost for eachs tate
    best_g = {start: 0}

    # A* loop, pop from queue to test for goal
    while not fringe.isEmpty():
        state, path, g = fringe.pop()

        # Skip if we’ve already found a cheaper way to state
        if g > best_g.get(state, float('inf')):
            continue

        # Test current sate for goal 
        if problem.isGoalState(state):
            return path

        # Relax edges
        for succ, action, stepCost in problem.getSuccessors(state):
            new_g = g + stepCost

            # try to improve the costs of "g's" neighbors 
            if new_g < best_g.get(succ, float('inf')):
                best_g[succ] = new_g
                f = new_g + heuristic(succ, problem)
                fringe.push((succ, path + [action], new_g), f)

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
