import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from common.import_data import ImportData


if __name__ == "__main__":
    data_set = ImportData()
    x_train: np.ndarray = data_set.import_train_data()
    x_test: np.ndarray = data_set.import_test_data()
    y_train: np.ndarray = data_set.import_columns_train(np.array(['quality']))
    y_test: np.ndarray = data_set.import_columns_test(np.array(['quality']))

    NN = MLPClassifier(solver='lbfgs', alpha=1e-2, hidden_layer_sizes=(150, 11), random_state=1).fit(x_train,
                                                                                                     y_train.ravel())
    print(NN.predict(x_test))
    print(round(NN.score(x_test, y_test.ravel()), 4))

