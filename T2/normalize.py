from sklearn.preprocessing import MinMaxScaler
import numpy as np

# # Prepare your dataset (2D array)
# data = np.array([[1, 2, 3, 10],
#                  [4, 5, 6, 20],
#                  [7, 8, 9, 30]])

# data = np.array([[1500,10,0,2000, 4],
#                  [1090,10,58,2000, 5],
#                  [1080,10,58,2000, 7],
#                  [1070,10,58,2000, 9]])



# # Separate the last column
# last_column = data[:, -1]

# #  Normalize the remaining columns (0 to 1)
# scaler = MinMaxScaler(feature_range=(0, 1))
# # normalized_data = scaler.fit_transform(data[:, :-1])
# normalized_data = scaler.fit_transform(data)


# # Normalize the last column (-1 to 1)
# # last_column_scaler = MinMaxScaler(feature_range=(-1, 1))
# # normalized_last_column = last_column_scaler.fit_transform(last_column.reshape(-1, 1))

# # Combine the normalized data and the normalized last column
# # normalized_data = np.concatenate((normalized_data, normalized_last_column), axis=1)

# print("Original data:")
# print(data)
# print("Normalized data (0 to 1, except last column -1 to 1):")
# print(normalized_data)

# import random

# random_number = random.random()
# print(random_number)

# import random

# random_integer = random.randrange(-1, 2)
# print(random_integer)


# array1 = [[1, 2, 3],
#           [3, 2, 1]]

# array2 = [4, 5]

# for i in range(len(array2)):
#     array1[i] += array2[i]

# print(array1)


# array1 = [[1, 2, 3],
#           [4, 5, 6]]

# array2 = [10, 20]

#aqui ficou certo

# array1 = [[1500,10,0,2000],
#             [1090,10,58,2000],
#             [1080,10,58,2000],
#             [1070,10,58,2000]]

# array2 = [-1 , -1 , -1 , -1]

# for i in range(len(array1)):
#     array1[i].append(array2[i])

# print(array1)


# states = [[1500,10,0,2000],
#             [1090,10,58,2000],
#             [1080,10,58,2000],
#             [1070,10,58,2000]]

# targets = [-1 , -1 , -1 , -1]


# def joining_state_target(states, targets):
#     #states_targets = [ [] for _ in range(len(states))]
#     print('________________')
#     print(states)
#     print('________________')


#     states_targets = states.copy()

#     for i in range(len(states)):
#         states_targets[i].append(targets[i])

#     print(states_targets)
#     print('________________')
#     print(states)
#     #return states_targets

# joining_state_target(states, targets)


# def joining_state_target(states, targets):
#     states_targets = []

#     for i in range(len(states)):
#         subarray = states[i].copy()  
#         subarray.append(targets[i])  
#         states_targets.append(subarray)  

#     # print(states_targets)
#     # print('________________')
#     # print(states)
#     return states_targets

# # states = [[1, 2, 3], [4, 5, 6]]
# # targets = [10, 20]

# joining_state_target(states, targets)


# array = [[1, 2, 3],
#          [4, 5, 6],
#          [7, 8, 9]]

# last_elements = []
# captured_array = []

# for subarray in array:
#     last_element = subarray[-1]  # Último elemento do subarray
#     last_elements.append(last_element)

#     captured_subarray = subarray[:-1]  # Captura todos os elementos, exceto o último
#     captured_array.append(captured_subarray)

# print(last_elements)
# print('--------------------')
# print(captured_subarray)



# def capture_elements(array):
#     captured_array = []
#     last_elements = []


#     for subarray in array:
#         captured_subarray = subarray[:-1]  # Captura todos os elementos, exceto o último
#         captured_array.append(captured_subarray)

#         last_element = subarray[-1]  # Último elemento do subarray
#         last_elements.append(last_element)

#     return captured_array, last_elements

# array = [[1, 2, 3],
#          [4, 5, 6],
#          [7, 8, 9]]

# captured_elements, last_elements = capture_elements(array)
# print(captured_elements)
# print('-------------------')
# print( last_elements)


# def separating_states_targets(states_targets):
#     states = []
#     targets = []

#     for subarray in states_targets:
#         state = subarray[:-1]  
#         states.append(state)

#         target = subarray[-1] 
#         targets.append(target)
#     print('-----states_targerts--------------')
#     print(states_targets)
#     print('-------------------')

#     return states, targets

# states_targets = [[1500, 10, 0, 2000, -1], 
#                   [1090, 10, 58, 2000, -1],
#                   [1080, 10, 58, 2000, -1], 
#                   [1070, 10, 58, 2000, -1]]

# states, targets = separating_states_targets(states_targets)
# print(states)
# print('-------------------')
# print(targets)



# from sklearn.preprocessing import MinMaxScaler

# def normalize_data(data):
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     normalized_data = scaler.fit_transform(data)

#     return normalized_data

# # Exemplo de uso:
# # data = [[10], [20], [30], [40], [50]]


# data = np.array([[1500,10,0,2000, 4],
#                  [1090,10,58,2000, 5],
#                  [1080,10,58,2000, 7],
#                  [1070,10,58,2000, 9]])


# normalized_data = normalize_data(data)
# print(normalized_data)


# from sklearn.preprocessing import MinMaxScaler
# import numpy as np

# array = np.array([1, 2, 3, 4, 5])
# array = array.reshape(-1, 1)

# scaler = MinMaxScaler()
# normalized_array = scaler.fit_transform(array)

# print(normalized_array)


import numpy as np

def normalize_array(array):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array

# Exemplo de uso
array = [1, 2, 3, 4, 5]
normalized_array = normalize_array(array)
print(normalized_array)