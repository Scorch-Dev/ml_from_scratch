
import numpy as np
import os

PWD = os.path.realpath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PWD, 'data')
WINE_DATA_FILE = os.path.join(DATA_DIR, 'wine.data')

def load_wine_data():
    """
    Loads the labels and data for the wine datset

    @returns the a tuple (data, labels)
        where the data as an np array (shape=(178,13))
        and the labels are an np array (shape=(178,))
    """
    dataRaw = np.genfromtxt(WINE_DATA_FILE, delimiter=',')
    data = dataRaw[:,1:]
    labels = dataRaw[:,0]
    return (data,labels)

def normalize_columns(mat):
    """
    Helper function saves us a line or two
    and increases clarity

    @returns the column-normalized [0,1] matrix
    """
    return np.divide(mat, np.std(mat, axis=0))

def subtract_column_mean(mat):
    mean = np.mean(mat, axis=0)
    ret = mat - mean
    return ret

if __name__ == "__main__":
    """
    sanity check functionality
    """

    # test loading and filesystem is all good
    X, y = load_wine_data()
    assert(X is not None)
    assert(y is not None)

    # check normalizing wasn't bugged
    xNormed = normalize_columns(X)
    assert(np.isclose(np.sum(xNormed, axis=0), np.ones(X.shape[1])).all())