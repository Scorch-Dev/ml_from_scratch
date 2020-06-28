
import numpy as np

def init_model(data, numStates):
    """
    Initializes the model using the data
    to create a transition matrix,
    where a transition probability from state i to 
    state j is given by the j_the column of the i_th row. 
    The matrix is of size (numStates,numStates).

    @param data: the list of team names
    @param nStates: the number of states in the transition matrix
    @returns an empty transition matrix as described above (shape = (numStates,numStates))
    """
    M = np.ndarray((numStates,numStates))
    
    for row in data:

        aIndex  = row[0].astype(np.int64)
        aPoints = row[1]
        bIndex  = row[2].astype(np.int64)
        bPoints = row[3]

        # calc some intermediate values to update mat
        aWins = int(aPoints > bPoints)
        bWins = int(bPoints > aPoints)
        totalPoints = aPoints + bPoints
        aPointsFrac = aPoints / totalPoints
        bPointsFrac = bPoints / totalPoints

        # update mat without conditionals for speed
        M[aIndex, aIndex] = M[aIndex, aIndex] + aWins + aPointsFrac # a -> a
        M[aIndex, bIndex] = M[aIndex, bIndex] + bWins + bPointsFrac # a -> b
        M[bIndex, bIndex] = M[bIndex, bIndex] + bWins + bPointsFrac # b -> b
        M[bIndex, aIndex] = M[bIndex, aIndex] + aWins + aPointsFrac # b -> a

    # now normalize
    M = np.apply_along_axis(lambda row: np.divide(row, np.sum(row)), 1, M)

    return M

def rank_teams(M, numTimeSteps):
    """
    Creates a team ranking by generating a probability
    distribution using the transition matrix for t
    time steps out.
    
    @param M: the n by n transition matrix created from init_model()
    @param numTimeSteps: how many time steps as an integer to predict out
    @returns the np array of team ranking scores. E.g. the ith index
        of this array indicates the given "ranking score" of the ith team
    """
    w0 = np.ones((1,M.shape[0])) / M.shape[0] # uniform distribution
    wt = np.matmul(w0, np.linalg.matrix_power(M, numTimeSteps))
    return wt.flatten()

def rank_teams_stationary(M):
    """
    Creates a team ranking by generating a probability
    distribution using the transition matrix at
    timestep t = infiniti by taking the stationary distribution
    of the transpose of the matrix

    @param M: the n by n transition matrix created from init_model()
    @returns the np array of team ranking scores. E.g. the ith index
        of this array indicates the given "ranking score" of the ith team
    """

    u, v = np.linalg.eig(M.transpose())
    v1 = v[:,np.isclose(u, 1)]
    v1 = v1[:,0]

    wStationary = (v1 / np.sum(v1)).real
    return wStationary