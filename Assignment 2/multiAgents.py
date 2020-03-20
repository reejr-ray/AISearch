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
import random
import util

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
        newx, newy = newPos
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        '''
        print("*******************************\n")
        print("Evaluation of", action, ": ")
        print(successorGameState)
        print(newPos)
        print(newFood)
        for ghost in newGhostStates:
            print(str(ghost))
            ghostPos = ghost.getPosition()
            ghostDis = manhattanDistance(newPos, ghostPos)
            print("distance to ghost:", ghostDis)
        #print(newScaredTimes)
        print(successorGameState.getScore())
        print("\n*******************************")
        '''
        # gets high score of 1200. not bad, but doesn't take into account the powerups.
        score = successorGameState.getScore()

        # test to see if ghost is close to the pacman & if ghost is intending to move into pacman
        for ghost in newGhostStates:
            ghostPos = ghost.getPosition()
            dir = ghost.getDirection()
            ghostDis = 3 - manhattanDistance(newPos, ghostPos)
            if ghostDis <= 2:
                # pacman has to pay attention to the ghost's direction
                if dir != action:
                    # collision imminent
                    score -= 6 ** ghostDis
                else:
                    score -= 5 ** ghostDis
            else:
                score -= 4 ** ghostDis

        for point in newFood.asList():
            foodDis = 1 / manhattanDistance(newPos, point)
            score += foodDis

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions()

        highScore = float('-inf')  # initialize the value to neg. infinity
        bestAction = None
        for action in legalMoves:
            if action == None:  # prevents returning Nonetype
                bestAction = action
            successorGameState = gameState.generateSuccessor(0, action)
            '''   1 depth = 1 action looked at
                  1 index = ghost's turn (since we start in depth 1 instead of at top)
                  call minimizing function because 
            '''
            # print("starting minimax at depth of", self.depth)
            newScore = self.getMin(successorGameState, self.depth, 1)  # 1 depth, 1 index = ghost
            if newScore > highScore:
                highScore = newScore
                bestAction = action
        # after all actions have been exhausted, return the best action.
        # print("Best action is", bestAction, "with a highscore of", highScore)
        return bestAction

    def getMax(self, gameState, depth, index):
        '''
        returns the maximum value - In this case, used for PACMAN
        '''
        # initialize the value to neg. infinity
        value = float('-inf')
        # if no valid moves left, return score
        if depth == 0 or len(gameState.getLegalActions(index)) == 0:  # getLegalActions already calls isWin() and isLose()
            return self.evaluationFunction(gameState)
        # pacman's turn
        for action in gameState.getLegalActions(index):
            # TODO call getMin wth index incremented, same depth
            successor = gameState.generateSuccessor(index, action)
            # print("Calling getMin() on", action, "keeping depth at", depth, "and setting index to", index + 1)
            childVal = self.getMin(successor, depth, index + 1)
            value = max(value, childVal)
        return value

    def getMin(self, gameState, depth, index):  # index needed because ghosts use this function
        '''
            returns the minimum value - used for GHOSTS or min players
        '''
        # if no valid moves left, return score
        if depth == 0 or len(gameState.getLegalActions(index)) == 0:  # getLegalActions already calls isWin() and isLose()
            return self.evaluationFunction(gameState)

        ''' need a for-loop for each player to check all positions for best action
            on that minimizing player
        '''
        # first to almost last ghost's turn
        if index < gameState.getNumAgents() - 1:  # provided that index is always >= 1 in getMin
            value = float('inf')  # initialize val to pos. infinity
            # TODO call getMin with incremented index, same depth
            for action in gameState.getLegalActions(index): # need 2 for loops to check all positions for
                #generate a successor for that action
                successor = gameState.generateSuccessor(index, action)
                # check the value of the child
                # print("Calling getMin() on", action, "keeping depth at", depth, "and setting index to", index+1)
                childVal = self.getMin(successor, depth, index + 1)
                #take the minimum of the current low and child's val
                value = min(value, childVal)
            return value

        # last ghost's turn -> pacman's turn
        else:
            value = float('inf')
            # TODO call getMax with index 0, increased depth
            for action in gameState.getLegalActions(index):
                successor = gameState.generateSuccessor(index, action)
                # print("Calling getMax() on", action, "with a depth of", depth+1, "and setting index to", 0)
                childVal = self.getMax(successor, depth - 1, 0)
                value = min(value, childVal)
            return value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions()

        highScore = float('-inf')  # initialize the value to neg. infinity
        bestAction = None
        alpha = float('-inf')
        beta = float('inf')
        for action in legalMoves:
            if action == None:  # prevents returning Nonetype
                bestAction = action
            successorGameState = gameState.generateSuccessor(0, action)
            '''   1 depth = 1 action looked at
                  1 index = ghost's turn (since we start in depth 1 instead of at top)
                  call minimizing function because 
            '''
            # print("starting minimax at depth of", self.depth)
            newScore = self.getMin(successorGameState, self.depth, 1, alpha, beta)  # 1 depth, 1 index = ghost
            # same as max, but grabbing the best action
            if newScore > highScore:
                highScore = newScore
                bestAction = action
            alpha = max(alpha, newScore)
            if beta < alpha:
                break
        # after all actions have been exhausted, return the best action.
        # print("Best action is", bestAction, "with a highscore of", highScore)
        return bestAction

    def getMax(self, gameState, depth, index, alpha, beta):
        '''
        returns the maximum value - In this case, used for PACMAN
        '''
        # initialize the value to neg. infinity
        value = float('-inf')
        # if no valid moves left, return score
        if depth == 0 or len(
                gameState.getLegalActions(index)) == 0:  # getLegalActions already calls isWin() and isLose()
            return self.evaluationFunction(gameState)
        # pacman's turn
        for action in gameState.getLegalActions(index):
            # TODO call getMin wth index incremented, same depth
            successor = gameState.generateSuccessor(index, action)
            # print("Calling getMin() on", action, "keeping depth at", depth, "and setting index to", index + 1)
            childVal = self.getMin(successor, depth, index + 1, alpha, beta)
            value = max(value, childVal)
            alpha = max(alpha, childVal)
            if beta < alpha:
                break
        return value

    def getMin(self, gameState, depth, index, alpha, beta):  # index needed because ghosts use this function
        '''
            returns the minimum value - used for GHOSTS or min players
        '''
        # if no valid moves left, return score
        if depth == 0 or len(
                gameState.getLegalActions(index)) == 0:  # getLegalActions already calls isWin() and isLose()
            return self.evaluationFunction(gameState)

        ''' need a for-loop for each player to check all positions for best action
            on that minimizing player
        '''
        # first to almost last ghost's turn
        if index < gameState.getNumAgents() - 1:  # provided that index is always >= 1 in getMin
            value = float('inf')  # initialize val to pos. infinity
            # TODO call getMin with incremented index, same depth
            for action in gameState.getLegalActions(index):  # need 2 for loops to check all positions for
                # generate a successor for that action
                successor = gameState.generateSuccessor(index, action)
                # check the value of the child
                # print("Calling getMin() on", action, "keeping depth at", depth, "and setting index to", index+1)
                childVal = self.getMin(successor, depth, index + 1, alpha, beta)
                # take the minimum of the current low and child's val
                value = min(value, childVal)
                beta = min(beta, childVal)
                if beta < alpha:
                    break
            return value

        # last ghost's turn -> pacman's turn
        else:
            value = float('inf')
            # TODO call getMax with index 0, increased depth
            for action in gameState.getLegalActions(index):
                successor = gameState.generateSuccessor(index, action)
                # print("Calling getMax() on", action, "with a depth of", depth+1, "and setting index to", 0)
                childVal = self.getMax(successor, depth - 1, 0, alpha, beta)
                value = min(value, childVal)
                beta = min(beta, childVal)
                if beta < alpha:
                    break
            return value

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
        legalMoves = gameState.getLegalActions()

        highScore = float('-inf')  # initialize the value to neg. infinity
        bestAction = None
        for action in legalMoves:
            if action == None:  # prevents returning Nonetype
                bestAction = action
            successorGameState = gameState.generateSuccessor(0, action)
            '''   1 depth = 1 action looked at
                  1 index = ghost's turn (since we start in depth 1 instead of at top)
                  call minimizing function because 
            '''
            newScore = self.getMin(successorGameState, self.depth, 1)  # 1 depth, 1 index = ghost
            if newScore > highScore:
                highScore = newScore
                bestAction = action
        return bestAction

    def getMax(self, gameState, depth, index):  # stays the same as minimax
        """
        returns the maximum value - In this case, used for PACMAN
        """
        value = float('-inf')
        if depth == 0 or len(
                gameState.getLegalActions(index)) == 0:  # getLegalActions already calls isWin() and isLose()
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(index):
            # TODO call getMin wth index incremented, same depth
            successor = gameState.generateSuccessor(index, action)
            childVal = self.getMin(successor, depth, index + 1)
            value = max(value, childVal)
        return value

    def getMin(self, gameState, depth, index):  # index needed because ghosts use this function
        """
            returns the minimum value - used for GHOSTS or min players
        """
        # getLegalActions already calls isWin() and isLose()
        if depth == 0 or len(gameState.getLegalActions(index)) == 0:
            return self.evaluationFunction(gameState)
        ''' 
            need a for-loop for each player to check all positions for best action
            on that minimizing player
        '''

        if index < gameState.getNumAgents() - 1:  # provided that index is always >= 1 in getMin
            value = 0
            # TODO call getMin with incremented index, same depth
            for action in gameState.getLegalActions(index):
                successor = gameState.generateSuccessor(index, action)
                # print("Calling getMin() on", action, "keeping depth at", depth, "and setting index to", index+1)
                childVal = self.getMin(successor, depth, index + 1)
                value += childVal
            return value / float(len(gameState.getLegalActions(index)))

        # last ghost's turn -> pacman's turn
        else:
            value = 0
            # TODO call getMax with index 0, increased depth
            for action in gameState.getLegalActions(index):
                successor = gameState.generateSuccessor(index, action)
                # print("Calling getMax() on", action, "with a depth of", depth+1, "and setting index to", 0)
                childVal = self.getMax(successor, depth - 1, 0)
                value += childVal
            return value / float(len(gameState.getLegalActions(index)))


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    highScore = 0
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    pacPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    capsules = currentGameState.getCapsules()  # get all the capsules.
    numFood = len(newFood.asList())
    numCapsule = len(capsules)
    ghostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    # gets high score of 1200. not bad, but doesn't take into account the powerups.
    score = successorGameState.getScore()

    # test to see if ghost is close to the pacman & if ghost is intending to move into pacman
    for ghost in newGhostStates:
        ghostPos = ghost.getPosition()
        dir = ghost.getDirection()
        ghostDis = 3 - manhattanDistance(newPos, ghostPos)
        if ghostDis <= 2:
            # pacman has to pay attention to the ghost's direction
            if dir != action:
                # collision imminent
                score -= 6 ** ghostDis
            else:
                score -= 5 ** ghostDis
        else:
            score -= 4 ** ghostDis

    for point in newFood.asList():
        foodDis = 1 / manhattanDistance(newPos, point)
        score += foodDis

    return score

# Abbreviation
better = betterEvaluationFunction
