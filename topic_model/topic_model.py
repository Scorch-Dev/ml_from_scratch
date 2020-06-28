
import numpy as np

def init_model(v, n, k):
    """
    Initializes our NMF topic model
    using the input data X. Each
    row in X is a "bag of words" data
    entry, where the jth entry of the ith
    row is the number of times word j appears
    in the bag of words i.

    @param v: the vocab size
    @param n: number of documents
    @param k: the rank of the word NMF embedding
    @returns a model tuple (W,H) for the NMF, initialized
        to uniform(1,2). W is a numpy array (shape=(n,k))
        and H is a numpy array (shape=(k,v))
    """
    W = np.random.uniform(1.0, 2.0, size=(v,k))
    H = np.random.uniform(1.0, 2.0, size=(k,n))
    return (W,H)

def train_model(model, data, epochs):
    """
    trains an NMF model using an initialized
    model, and updates the model in place.

    @param model: a (W,H) tuple returned from init_model
    @param data: the n by v input data where
        the jth column of the ith row indicates
        the number of items word j appeared in example i
    @param epochs: how many training updates to perform
    """

    W, H = model
    losses = np.ndarray((epochs,))

    # some temporary arrays to avoid reallocation all the time
    xHolder = np.ndarray(data.shape)
    wTHolder = np.ndarray((W.T.shape))
    hTHolder = np.ndarray((H.T.shape))

    norm1D = lambda v : v / np.sum(v)

    for t in range(epochs):

        # update H
        approx = np.matmul(W,H) + 10e-16
        xHolder[:,:] = np.divide(data, approx)                # x / xHat
        wTHolder[:,:] = np.apply_along_axis(norm1D, 1, W.T)   # norm rows of W.T
        H[:,:] = np.multiply(H, np.matmul(wTHolder, xHolder))
        
        # update W
        approx = np.matmul(W,H) + 10e-16
        xHolder[:,:] = np.divide(data, approx)                # x / xHat
        hTHolder[:,:] = np.apply_along_axis(norm1D, 0, H.T)   # norm cols of H.T
        W[:,:] = np.multiply(W, np.matmul(xHolder, hTHolder))

        # calc objective function
        losses[t] = calc_objective(model, data)

    return losses
    
def calc_objective(model, data):
    """
    Calculates the loss using the divergence objective function
    
    @param model: the (W,H) matrix given by init_model()
    @param data: the n by vocabSize input data
    """
    xHat = np.matmul(model[0], model[1]) + 10e-16
    logXHat = np.log(xHat)

    divergence = (np.multiply(data, logXHat) - xHat)
    return (-1 * np.sum(divergence))