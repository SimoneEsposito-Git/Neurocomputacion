# pylint: disable=redefined-outer-name, trailing-whitespace, pointless-string-statement, invalid-name, missing-docstring, too-few-public-methods, no-self-use, too-many-arguments, too-many-locals, too-many-statements, too-many-branches, too-many-boolean-expressions, too-many-instance-attributes, too-many-ancestors, too-many-public-methods, too-many-lines, too-many-arguments, too-many-branches, too-many-locals, too-many-statements, too-many-boolean-expressions, too-many-instance-attributes, too-many-ancestors, too-many-public-methods, too-many-lines

import reader

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

    def disparar(self, umbral):
        if self.valor < -umbral:
            self.valor = -1
        elif self.valor > umbral:
            self.valor = 1
        else:
            self.valor = 0
        # print(self.valor)

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

    def disparar(self, umbral):
        for neurona in self.neuronas:
            neurona.disparar(umbral)
    
    def propagar(self):
        for neurona in self.neuronas:
            neurona.propagar()
    
    def imprimir(self):
        for neurona in self.neuronas:
            neurona.imprimir()

class Perceptron:
    def __init__(self, input_, output, umbral=0.2):
        self.input = input_
        self.output = output
        self.umbral = umbral

    def disparar(self, umbral):
        self.output.disparar(umbral)
    
    def propagar(self):
        self.input.propagar()
    
    def imprimir(self):
        self.input.imprimir()
        self.output.imprimir()
    
    def cambiar_pesos(self, out, expected, alpha):
        if self.input:
            for neuron in self.input.neuronas:
                conexion = neuron.get_connectado(out)
                new_peso = conexion.peso + alpha * expected * neuron.valor
                conexion.cambiar_peso(new_peso)

    def fit(self, X_train, y_train, epochs=10, alpha=1, verbose = False, ecm = False):
        ecms = []
        if len(X_train[0]) != len(self.input.neuronas) - 1 or len(y_train[0]) != len(self.output.neuronas):
            raise ValueError("Dimensiones de entrada y salida no coinciden con la red")
        for epoch in range(epochs):
            if verbose:
                print(f"---EPOCH {epoch+1}---")
            for x, y in zip(X_train, y_train):      
                for neuron, val in zip(self.input.neuronas, list(x)):
                    neuron.inicializar(val)
                for neuron in self.output.neuronas:
                    neuron.valor = 0
                self.propagar()
                self.disparar(self.umbral)
                # self.cambiar_pesos(y, alpha)
                for i, neuron in enumerate(self.output.neuronas):
                    if neuron.valor != y[i]:     
                        self.cambiar_pesos(neuron, y[i], alpha)
                
                if verbose:
                    self.imprimir()
                    for i, neuron in enumerate(self.input.neuronas):
                        for j, conexion in enumerate(neuron.conexiones):
                            print(f'w_{i},{j}: {round(conexion.peso, 2)}', end="| ")
                    print("\n")
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
            self.input.propagar()
            self.output.disparar(self.umbral)
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

    def accuracy(self, y_pred, y_true):
        correct_predictions = 0
        total_predictions = 0

        for true, pred in zip(y_true, y_pred):
            for t, p in zip(true, pred):
                if t == p:
                    correct_predictions += 1
                total_predictions += 1

        return correct_predictions / total_predictions
    
    def print_weights(self):
        for n in self.input.neuronas:
            print(n.nombre + "'s weight = " + str(n.conexiones[0].peso))


def new_perceptron(n_in, n_out, umbral=0.2):
    input_layer = Capa()
    output_layer = Capa()

    for i in range(n_in):
        input_layer.anadir_neurona(Neurona('input_' + str(i)))
    for i in range(n_out):
        output_layer.anadir_neurona(Neurona('output_' + str(i)))
    
    input_layer.anadir_neurona(Neurona('bias'))

    input_layer.conectar(output_layer, 0)

    return Perceptron(input_layer, output_layer, umbral)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X, y = reader.leer2("NEURO-prac/P1/Data/xor.txt")
    '''
    and_perception = new_perceptron(2, 1)
    and_perception.fit(X, y, epochs = 10, alpha=1, verbose = False, ecm = False)
    and_perception.print_weights()
    '''

    X_train, y_train, X_test, y_test = reader.leer1("NEURO-prac/P1/Data/problema_real1.txt", 0.8)
    red_bin = new_perceptron(2, 1)
    red_bin.fit(X, y, epochs = 10, alpha=1, verbose = False, ecm = False)

    # Initialize lists to hold ECM and accuracy values for each threshold
    ecm_lists = []
    accuracy_lists = []

    # Threshold values to iterate over
    umbrales = [0.1, 0.3, 0.5]

    for umbral in umbrales:
        ecm_list = []
        accuracy_list = []

        for i in range(1, 21):
            red = new_perceptron(9, 2, umbral) 
            red.fit(X_train, y_train, epochs=i, alpha=1, verbose=False, ecm=False)
            accuracy_list.append(red.accuracy(red.predict(X_test), y_test))
            ecm_list.append(red.ecm(red.predict(X_test), y_test))
        
        ecm_lists.append(ecm_list)
        accuracy_lists.append(accuracy_list)

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot ECM for each threshold
    plt.subplot(1, 2, 1)
    for i, umbral in enumerate(umbrales):
        plt.plot(ecm_lists[i], marker='o', linestyle='-', label=f'Umbral {umbral}')
    plt.title('ECM por Época para diferentes Umbrales')
    plt.xlabel('Época')
    plt.ylabel('ECM')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy for each threshold
    plt.subplot(1, 2, 2)
    for i, umbral in enumerate(umbrales):
        plt.plot(accuracy_lists[i], marker='o', linestyle='-', label=f'Umbral {umbral}')
    plt.title('Precisión por Época para diferentes Umbrales')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

