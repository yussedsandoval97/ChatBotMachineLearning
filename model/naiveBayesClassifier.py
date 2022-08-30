class NaiveBayesClassifier:

    def __init__(self, X, y):

        '''
        X e y denotan las características y las etiquetas de destino respectivamente
        '''
        self.X, self.y = X, y

        self.N = len(self.X) # Tamaño del conjunto de entrenamiento

        self.dim = len(self.X[0]) # Dimensión del vector de características

        self.attrs = [[] for _ in range(self.dim)] # Aquí almacenaremos las columnas del conjunto de entrenamiento.

        self.output_dom = {} # Clases de salida con el número de ocurrencias en el conjunto de entrenamiento. En este caso solo tenemos 2 clases

        self.data = [] # To store every row [Xi, yi]


        for i in range(len(self.X)):
            for j in range(self.dim):
                # si nunca hemos visto este valor para este atributo antes,
                # luego lo agregamos a la matriz attrs en la posición correspondiente
                if not self.X[i][j] in self.attrs[j]:
                    self.attrs[j].append(self.X[i][j])

            # si nunca hemos visto esta clase de salida antes,
            # luego lo agregamos a output_dom y contamos una ocurrencia por ahora
            if not self.y[i] in self.output_dom.keys():
                self.output_dom[self.y[i]] = 1
            # de lo contrario, incrementamos la ocurrencia de esta salida en el conjunto de entrenamiento en 1
            else:
                self.output_dom[self.y[i]] += 1
            # almacenar la fila
            self.data.append([self.X[i], self.y[i]])



    def classify(self, entry):

        solve = None # Resultado final
        max_arg = -1 # máximo parcial

        for y in self.output_dom.keys():

            prob = self.output_dom[y]/self.N # P(y)

            for i in range(self.dim):
                cases = [x for x in self.data if x[0][i] == entry[i] and x[1] == y] # all rows with Xi = xi
                n = len(cases)
                prob *= n/self.N # P *= P(Xi = xi)

            # si tenemos una probabilidad mayor para esta salida que el máximo parcial ...
            if prob > max_arg:
                max_arg = prob
                solve = y

        return solve