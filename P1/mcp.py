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
    def __init__(self, umbral: float, tipo: str, nombre: str):
        self.umbral = umbral
        self.valor = 0.0
        self.tipo = tipo
        self.salida_activa = 1.0
        self.salida_inactiva = -1.0 if self.tipo == 'perceptron' else 0.0
        self.f_x = 0.0
        self.conexiones = []
        self.nombre = nombre

    def conectar(self, neurona: 'Neurona', peso: float):
        self.conexiones.append(Conexion(peso, neurona))

    def inicializar(self, valor: float):
        self.valor = valor

    def disparar(self):
        if self.tipo == 'mcp':
            if self.valor >= self.umbral:
                self.f_x = self.salida_activa
            else:
                self.f_x = self.salida_inactiva
        elif self.tipo == 'directo':
            self.f_x = self.valor
        elif self.tipo == 'perceptron':
            if self.valor > self.umbral:
                self.f_x = self.salida_activa
            elif self.valor < -self.umbral:
                self.f_x = self.salida_inactiva
            else:
                self.f_x = 0.0
        self.valor = 0.0

    def propagar(self):    
        for conexion in self.conexiones:
            conexion.neurona.valor += self.f_x*conexion.peso

    def imprimir(self):
        print(self.nombre + " = " + str(self.f_x) + " | ", end="")

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

    def disparar(self):
        for neurona in self.neuronas:
            neurona.disparar()
    
    def propagar(self):
        for neurona in self.neuronas:
            neurona.propagar()
    
    def imprimir(self):
        for neurona in self.neuronas:
            neurona.imprimir()

class RedNeuronal:
    def __init__(self, tipo: str):
        self.capas = []
        self.tipo = tipo
        self.alpha = 0.4    #Learning rate

    def anadir_capa(self, capa: Capa):
        self.capas.append(capa)

    def disparar(self):
        for capa in self.capas:
            capa.disparar()
    
    def propagar(self):
        for capa in self.capas:
            capa.propagar()
    
    def imprimir(self):
        for capa in self.capas:
            capa.imprimir()
        print("\n")
    
    def set_alpha(self, val):
        self.alpha = val

    def cambiar_pesos(self, output, expected):
        if self.tipo == 'perceptron':
            if (output != expected):
                for capa in self.capas:
                    for neuron in capa.neuronas:
                        for conexion in neuron.conexiones:
                            if neuron.tipo == 'bias':
                                conexion.cambiar_peso(conexion.peso + self.alpha * (expected-output))
                            else:   
                                conexion.cambiar_peso(conexion.peso + self.alpha * (expected-output) * neuron.f_x)

    def entrenar(self, X, Y, epochs):
        for epoch in range(epochs):
            print("---EPOCH " + str(epoch) + "---")
            for x, y in zip(X, Y):      
                for neuron, val in zip(self.capas[0].neuronas, list(x)):
                    neuron.inicializar(val)
                untrained = True
                j = 0
                while untrained:
                    for i in range(0, len(self.capas)):
                        print("t=" + str(j) + "." + str(i), end=": ")
                        self.disparar()
                        self.propagar()
                        self.imprimir()
                    j += 1
                    if self.capas[len(self.capas)-1].neuronas[0].f_x != y:     #Checking output neuron, need to change for y>1
                        #add method to red to fetch output layer values
                        self.cambiar_pesos(self.capas[len(self.capas)-1].neuronas[0].f_x, y)
                    else:
                        untrained = False
    
    def test(self, X, Y):
        for x, y in zip(X, Y):      
            for neuron, val in zip(self.capas[0].neuronas, list(x)):
                neuron.inicializar(val)
            for i in range(0, len(self.capas)):
                print("t=" + str(i), end=": ")
                self.disparar()
                self.propagar()
                self.imprimir()
            print(self.capas[len(self.capas)-1].neuronas[0].f_x,  y)

if __name__ == '__main__':
    red = RedNeuronal('mcp')
    capa1 = Capa()
    capa2 = Capa()
    capa3 = Capa()
    x1 = Neurona(1, 'mcp', "x1")
    x2 = Neurona(1, 'mcp', "x2")
    x3 = Neurona(1, 'mcp', "x3")
    and1 = Neurona(2, 'mcp', "and1")
    and2 = Neurona(2, 'mcp', "and2")
    and3 = Neurona(2, 'mcp', "and3")
    or1 = Neurona(1, 'mcp', "or1")

    capa1.anadir_neurona(x1)
    capa1.anadir_neurona(x2)
    capa1.anadir_neurona(x3)
    capa2.anadir_neurona(and1)
    capa2.anadir_neurona(and2)
    capa2.anadir_neurona(and3)
    capa3.anadir_neurona(or1)

    x1.conectar(and1, 1)
    x1.conectar(and3, 1)
    x2.conectar(and1, 1)
    x2.conectar(and2, 1)
    x3.conectar(and2, 1)
    x3.conectar(and3, 1)

    and1.conectar(or1, 1)
    and2.conectar(or1, 1)
    and3.conectar(or1, 1)

    red.anadir_capa(capa1)
    red.anadir_capa(capa2)
    red.anadir_capa(capa3)

    #Simulating example in example, not a value of 2 is needed for and to activate
    x1vals = [0, 0, 0, 0, 1, 1, 1]
    x2vals = [0, 0, 1, 1, 0, 0, 0]
    x3vals = [0, 1, 0, 1, 0, 1, 0]

    for i in range(0, len(x1vals)-1):
        print("t=" + str(i), end=": ")
        red.disparar()
        x1.inicializar(x1vals[i])
        x2.inicializar(x2vals[i])
        x3.inicializar(x3vals[i])
        red.propagar()
        red.imprimir()
        
    
