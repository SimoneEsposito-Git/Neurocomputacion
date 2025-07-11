# pylint: disable=unspecified-encoding, redefined-outer-name, trailing-whitespace, pointless-string-statement, invalid-name, missing-docstring, too-few-public-methods, no-self-use, too-many-arguments, too-many-locals, too-many-statements, too-many-branches, too-many-boolean-expressions, too-many-instance-attributes, too-many-ancestors, too-many-public-methods, too-many-lines, too-many-arguments, too-many-branches, too-many-locals, too-many-statements, too-many-boolean-expressions, too-many-instance-attributes, too-many-ancestors, too-many-public-methods, too-many-lines

import numpy as np

def leer1(fichero_de_datos, por):
    with open(fichero_de_datos, 'r') as file:
        M, N = map(int, file.readline().strip().split())
        
        data = np.loadtxt(file, dtype=float)
    
    # Split the data into features (X) and labels (y)
    X, y = data[:, :M], data[:, M:]
    
    # Shuffle the dataset
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    
    # Determine the split index
    split_idx = int(X.shape[0] * por)
    
    # Split into training and test sets
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train.tolist(), y_train.tolist(), X_test.tolist(), y_test.tolist()

def leer2(fichero_de_datos):
    with open(fichero_de_datos, 'r') as file:
        M, N = map(int, file.readline().strip().split())
        
        data = np.loadtxt(file, dtype=float)
    X, y = data[:, :M], data[:, M:]

    return X.tolist(), y.tolist()

def leer3(fichero_de_entrenamiento, fichero_de_test):
    with open(fichero_de_entrenamiento, 'r') as file:
        M, N = map(int, file.readline().strip().split())
        
        data = np.loadtxt(file, dtype=float)
    X_train, y_train = data[:, :M], data[:, M:]

    with open(fichero_de_test, 'r') as file:
        M, N = map(int, file.readline().strip().split())
        
        data = np.loadtxt(file, dtype=float)
    X_test, y_test = data[:, :M], data[:, M:]

    return X_train.tolist(), y_train.tolist(), X_test.tolist(), y_test.tolist()

# print(leer2("Data/and.txt"))
    