"""
Tic Tac Toe Player
"""

import math
from copy import deepcopy

X = "X"
O = "O"
EMPTY = None

def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]

def player(board):
    """
    Returns player who has the next turn on a board.
    """
    number_of_X = sum (row.count(X) for row in board)
    number_of_O = sum (row.count(O) for row in board)
    if (number_of_O > number_of_X):
        return X
    return O

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    ans = []
    for i in [0,1,2]:
        for j in [0,1,2]:
            if board[i][j] == EMPTY:
                ans.append((i,j))
    return ans

def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if action not in actions(board):
        raise Exception("That action is not possible on this board.")
    newboard = deepcopy(board)
    newboard[action[0]][action[1]] = player(newboard)
    return newboard

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    to_check = []
    for i in [0,1,2]:
        to_check.append([board[i][0],board[i][1],board[i][2]])
        to_check.append([board[0][i],board[1][i],board[2][i]])
    to_check.append([board[0][0],board[1][1],board[2][2]])
    to_check.append([board[0][2],board[1][1],board[2][0]])
    for l in to_check:
        if all(x == X for x in l):
            return X
        if all(x == O for x in l):
            return O
    return None

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) == X or winner(board) == O:
        return True
    for i in [0,1,2]:
        for j in [0,1,2]:
            if board[i][j] == EMPTY:
                return False
    return True

def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    if winner(board) == O:
        return -1
    return 0

def max_value(board, score_needed_to_prune=1):
    """
    Auxiliary function for Minimax algorithm, works together with min_value. 
    Computes min_value's of all actions available on the board and returns the maximum.
    If during this iteration it encounters a min_value that is higher or equal than the given score needed to prune, 
    it immediately returns that min_value.
    With each call of min_value it passes the maximum encountered so far as the score needed to prune.
    """
    if terminal(board):
        return utility(board)
    ans = -1
    for action in actions(board):
        min_here = min_value(result(board, action), ans)
        if min_here >= score_needed_to_prune:
            return min_here
        ans = max(ans, min_here)
    return ans

def min_value(board, score_needed_to_prune=-1):    
    """
    Auxiliary function for Minimax algorithm, works together with max_value. 
    Computes max_value's of all actions available on the board and returns the minimum.
    If during this iteration it encounters a max_value that is lower or equal than the given score needed to prune, 
    it immediately returns that max_value.
    With each call of max_value it passes the minimum encountered so far as the score needed to prune.
    """
    if terminal(board):
        return utility(board)
    ans = 1
    for action in actions(board):
        max_here = max_value(result(board, action), ans)
        if max_here <= score_needed_to_prune:
            return max_here
        ans = min(ans, max_here )
    return ans

def minimax(board):
    """
    Returns an optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    ans = (0,0)
    if player(board) == X:
        utility = -2
        for action in actions(board):
            min_here = min_value(result(board,action))
            if min_here > utility:
                utility = min_here
                ans = action
        return ans
    # if player(board) == O:
    utility = 2
    for action in actions(board):
        max_here = max_value(result(board,action))
        if max_here < utility:
            utility = max_here
            ans = action
    return ans