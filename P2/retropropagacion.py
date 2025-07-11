# pylint: disable=redefined-outer-name, trailing-whitespace, pointless-string-statement, invalid-name, missing-docstring, too-few-public-methods, no-self-use, too-many-arguments, too-many-locals, too-many-statements, too-many-branches, too-many-boolean-expressions, too-many-instance-attributes, too-many-ancestors, too-many-public-methods, too-many-lines, too-many-arguments, too-many-branches, too-many-locals, too-many-statements, too-many-boolean-expressions, too-many-instance-attributes, too-many-ancestors, too-many-public-methods, too-many-lines

import reader
import math 
import numpy as np

class Conexion:
    def __init__(self, peso: float, neurona: 'Neurona'):
        self.peso = peso
        self.peso_anterior = 0.0
        self.neurona = neurona

    def cambiar_peso(self, new_peso):
        self.peso = new_peso

    def crear(self, peso: float, neurona: 'Neurona'):
        pass

    def liberar(self):
        pass

class Neurona:
    def __init__(self, nombre):
        self.nombre = nombre
        self.valor = 1.0
        self.f_x = 1.0
        self.conexiones = []

    def conectar(self, neurona: 'Neurona', peso: float):
        self.conexiones.append(Conexion(peso, neurona))

    def inicializar(self, valor: float):
        self.valor = valor
    
    def get_connectado(self, neurona: 'Neurona'):
        for conexion in self.conexiones:
            if conexion.neurona == neurona:
                return conexion
        return None

    def activar(self):
        #bipolar sigmoid
        # self.valor = 2 / (1 + math.e**(-self.valor)) - 1 
        new_val = 1 / (1 + math.e**(-self.valor))
        self.valor = new_val

    def propagar(self):    
        for conexion in self.conexiones:
            conexion.neurona.valor += self.valor*conexion.peso

    def imprimir(self):
        print(self.nombre + " = " + str(self.valor) + " | ", end="")

class Capa:
    def __init__(self):
        self.neuronas = []

    def anadir_neurona(self, neurona: Neurona):
        self.neuronas.append(neurona)

    def conectar(self, capa: 'Capa', modo_peso: int):
        for neurona in self.neuronas:
            for neurona2 in capa.neuronas:
                neurona.conectar(neurona2, modo_peso)

    def conectar_neurona(self, neurona: Neurona, modo_peso: int):
        for neurona2 in self.neuronas:
            neurona.conectar(neurona2, modo_peso)

    def activar(self):
        for neurona in self.neuronas:
            neurona.activar()
    
    def propagar(self):
        for neurona in self.neuronas:
            neurona.propagar()
    
    def imprimir(self):
        for neurona in self.neuronas:
            neurona.imprimir()

class Retropropagacion:
    def __init__(self, input_,capa_, output_):
        self.input = input_
        self.output = output_
        self.capa = capa_
    
    def propagar(self):
        self.input.propagar()
        self.capa.activar()
        self.capa.neuronas[-1].valor = 1 #bias
        self.capa.propagar()
        self.output.activar()
    
    def inicializar(self, x):    
        for neuron, val in zip(self.input.neuronas, list(x)):
            neuron.inicializar(val)
        for neuron in self.capa.neuronas:
            neuron.valor = 0
        self.capa.neuronas[-1].valor = 1 #bias
        for neuron in self.output.neuronas:
            neuron.valor = 0
    
    def imprimir(self):
        self.input.imprimir()
        self.capa.imprimir()
        self.output.imprimir()
    
    def backpropagation(self, y, alpha):
        delta = {}
        for k, output_neuron in enumerate(self.output.neuronas):
            # delta[output_neuron.nombre] = (y[k] - output_neuron.valor) * 0.5*(1 - output_neuron.valor**2)
            delta[output_neuron.nombre] = (y[k] - output_neuron.valor) * output_neuron.valor*(1 - output_neuron.valor)
            #print(f'd_{output_neuron.nombre}: {round(delta[output_neuron.nombre], 2)}', end="| ")
            for hidden_neuron in self.capa.neuronas:
                conexion = hidden_neuron.get_connectado(output_neuron)
                conexion.peso_anterior = conexion.peso
                conexion.peso += alpha * delta[output_neuron.nombre] * hidden_neuron.valor
                # print(f'w_{hidden_neuron.nombre},{output_neuron.nombre}: {round(conexion.peso, 4)}', end="| ")

        for j, neuron in enumerate(self.capa.neuronas):
            delta[neuron.nombre] = 0
            for conexion in neuron.conexiones:
                delta[neuron.nombre] += delta[conexion.neurona.nombre] * conexion.peso
            # delta[neuron.nombre] *= 0.5*(1 - neuron.valor**2)
            delta[neuron.nombre] *= neuron.valor*(1 - neuron.valor)
            # print(f'd_{neuron.nombre}: {round(delta[neuron.nombre], 2)}', end="| ")
            for neuron_in in self.input.neuronas:
                conexion = neuron_in.get_connectado(neuron)
                conexion.peso_anterior = conexion.peso
                conexion.peso += alpha * delta[neuron.nombre] * neuron_in.valor
                # print(f'w_{neuron_in.nombre},{neuron.nombre}: {round(conexion.peso, 4)}', end="| ")
        #print("\n")



    def fit(self, X_train, y_train, epochs=10, alpha=1, verbose = False, ecm = False):
        ecms = []
        if len(X_train[0]) != len(self.input.neuronas) - 1 or len(y_train[0]) != len(self.output.neuronas):
            raise ValueError("Dimensiones de entrada y salida no coinciden con la red")
        for epoch in range(epochs):
            if verbose:
                print(f"---EPOCH {epoch+1}---")
            for X, y in zip(X_train, y_train): 
                self.inicializar(X)
                self.propagar()
                self.backpropagation(y, alpha)

                if verbose:
                    self.imprimir()
            if ecm:
                ecms.append(self.ecm(self.predict(X_train), y_train))
        if ecm:
            return ecms
                
    def predict(self, X_test, verbose = False):
        outputs = []
        for x in X_test:
            for neuron, val in zip(self.input.neuronas, list(x)):
                neuron.inicializar(val)
            for neuron in self.output.neuronas:
                neuron.valor = 0
            self.propagar()
            # 1 if neuron.valor > threshold else 0
            
            outputs.append([neurona.valor for neurona in self.output.neuronas])
            if verbose:
                self.imprimir()
                print("\n")
        return outputs

    def ecm(self, y_pred, y_true):
        sum_squared_error = 0
        total_predictions = 0

        for true, pred in zip(y_true, y_pred):
            for t, p in zip(true, pred):
                sum_squared_error += (t - p) ** 2
                total_predictions += 1

        return sum_squared_error / total_predictions

    def accuracy(self, y_pred, y_true, threshold = 0.5):
        correct_predictions = 0
        total_predictions = 0

        for true, pred in zip(y_true, y_pred):
            for t, p in zip(true, pred):
                if p > threshold:
                    p = 1
                else:
                    p = 0
                if t == p:
                    correct_predictions += 1
                total_predictions += 1

        return correct_predictions / total_predictions
    
    def print_weights(self):
        for n in self.input.neuronas:
            print(n.nombre + "'s weight = " + str(n.conexiones[0].peso))
    



def new_perceptron(n_in, n_hidden, n_out):
    input_layer = Capa()
    hidden_layer = Capa()
    output_layer = Capa()

    for i in range(n_in):
        input_layer.anadir_neurona(Neurona('input_' + str(i)))
    for i in range(n_hidden):
        hidden_layer.anadir_neurona(Neurona('hidden_' + str(i)))
    for i in range(n_out):
        output_layer.anadir_neurona(Neurona('output_' + str(i)))
    
    input_layer.anadir_neurona(Neurona('bias'))
    hidden_layer.anadir_neurona(Neurona('bias'))
    input_layer.conectar(hidden_layer, 0)
    hidden_layer.conectar(output_layer, 0)

    return Retropropagacion(input_layer, hidden_layer, output_layer)

class Scaler:
    def __init__(self):
        self.mean = 0
        self.std = 1
    def _mean(self, X):
        n_feat = len(X[0])
        n_samp = len(X)

        mean = [0]*n_feat
        for row in X:
            for i, val in enumerate(row):
                mean[i] += val
        return [m/n_samp for m in mean]
    def _std(self, X):
        n_feat = len(X[0])
        n_samp = len(X)
        std = [0]*n_feat
        for row in X:
            for i, val in enumerate(row):
                std[i] += (val-self._mean(X)[i])**2
        return [math.sqrt(s/n_samp) for s in std]
    
    def fit(self, X):
        self.mean = self._mean(X)
        self.std = self._std(X)
        print(self.mean)
        print(self.std)
    def transform(self, X):
        # print([[(x-m)/s for x,m,s in zip(row, scaler.mean, scaler.std)] for row in X])
        return [[(x-m)/s for x,m,s in zip(row, scaler.mean, scaler.std)] for row in X]
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)

class ScalerNP:
    def __init__(self):
        self.mean = 0
        self.std = 1

    def fit(self,X):
        self.mean = np.mean(X, axis = 0)
        self.std = np.sqrt(np.mean((X-self.mean)**2,axis=0))
        print(self.mean)
        print(self.std)
    
    def transform(self, X):
        return ((X-self.mean)/self.std)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X = [[0,1]]
    y = [[1]]
    X_train, y_train, X_test, y_test = reader.leer1("NEURO-prac/P2/data/problema_real5.txt", 0.8)
    scaler = ScalerNP()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # print(X_test)
    accuracy = []
    accuracy_train = []
    n_epochs = 20
    for i in range(n_epochs):
        red_bin = new_perceptron(16, 2, 2)
        ecm = red_bin.fit(X_train, y_train, epochs = i, alpha=0.1, verbose = False, ecm = True)
        pred_train = red_bin.predict(X_train)
        pred_test = red_bin.predict(X_test)
        accuracy.append(red_bin.accuracy(pred_test, y_test))
        accuracy_train.append(red_bin.accuracy(pred_train, y_train))
    plt.plot(accuracy_train, label='train')
    plt.plot(accuracy,label='test')
    plt.xticks(range(n_epochs))
    plt.title('Accuracy per Epoch')
    plt.legend(['train','test'])
    # plot ecm in a separate plot
    plt.show()
    plt.plot(ecm)
    plt.title('MSE per epoch')
    plt.xticks(range(n_epochs))
    plt.show()
    # red_bin.predict(X, verbose = True)

