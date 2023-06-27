import math
from collections import Counter

# Função para calcular a distância euclidiana entre dois pontos
def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)

# Função KNN
def knn(train_set, test_instance, k):
    distances = []
    for train_instance in train_set:
        distance = euclidean_distance(train_instance[:-1], test_instance)
        distances.append((train_instance, distance))
    distances.sort(key=lambda x: x[1])  # Ordenar as distâncias em ordem crescente
    neighbors = [item[0] for item in distances[:k]]  # Obter os k vizinhos mais próximos
    classes = [neighbor[-1] for neighbor in neighbors]  # Obter as classes dos vizinhos
    count = Counter(classes)
    return count.most_common(1)[0][0]  # Retornar a classe mais comum

# Base de dados Iris
iris_data = [
    [5.1, 3.5, 1.4, 0.2, 'Iris-setosa'],
    [4.9, 3.0, 1.4, 0.2, 'Iris-setosa'],
    [6.2, 3.4, 5.4, 2.3, 'Iris-virginica'],
    # Adicione mais instâncias da base de dados aqui...
]

# Exemplo de teste
test_instance = [5.7, 2.9, 4.2, 1.3]

# Classificação usando KNN com k=3
k = 3
predicted_class = knn(iris_data, test_instance, k)
print("Classe prevista:", predicted_class)


https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/