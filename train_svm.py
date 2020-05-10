
import numpy as np

from load_data import load_wine_data, normalize_columns, subtract_column_mean
from svm import SVM

EPOCHS = 8

if __name__ == "__main__":

    # load data and train a model
    X, y = load_wine_data()
    y[y != 1] = -1
    
    #X = subtract_column_mean(X)
    #X = normalize_columns(X)

    model = SVM(X, y)
    model.train(EPOCHS)
    
    # print number of support vectors we found
    print('+1 support vector count:', len(model.alpha[np.isclose(model.alpha, 1.0) & (model.y ==  1)]))
    print('-1 support vector count:', len(model.alpha[np.isclose(model.alpha, 1.0) & (model.y == -1)]))
    #print(model.predict(X))