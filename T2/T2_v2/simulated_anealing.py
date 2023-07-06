import random
import time
import math
import random

#Calcula o size do state considerando os items. Não é exatamente o tamanho do state
#é a soma de cada posição do state vezes o valor qu está em cada posição dos items
def state_size(state, items):
    size = 0
    for i in range(len(state)):
        size += state[i] * items[i][1]
    return size

def evaluate_state(state, items):
    value = 0
    for i in range(len(state)):
        value += state[i] * items[i][0]
    return value

#Escolhe um state dos states aleatoriamente e retorna esse state escolhido
def random_state(states):
    index = random.randint(0,len(states)-1)
    return states[index]

#Muda o valor do state na posição position 
def change_state(state,position,value):
    if state[position] == 0 and value < 0:
        return []
    state[position] = state[position] + value
    return state

#Muda a probabiblidade considerando o value, best_value e t
def change_probability(value,best_value,t):
    p = 1/(math.exp(1)**((best_value-value)/t))
    r = random.uniform(0,1)
    if r < p:
        return True
    else:
        return False

def generate_neighborhood(max_size, items, state):
    neighborhood = []
    for i in range(len(state)):
        aux = state.copy()
        new_state = change_state(aux,i,1)
        if state_size (new_state, items) <= max_size:
            neighborhood.append(new_state)
    for i in range(len(state)):
        aux = state.copy()
        new_state = change_state(aux,i,-1)
        if new_state != [] and state_size (new_state, items) <= max_size:
            neighborhood.append(new_state)
    return neighborhood


def simulated_annealing(state,t,alfa,items,max_size,iter_max,max_time):
    solution = state
    max_value = evaluate_state(solution,items)
    start = time.process_time()
    end = 0
    
    while t >= 1 and end-start <= max_time:        
        
        for _ in range(iter_max):    
            neighborhood = generate_neighborhood(max_size,items,state)
            if neighborhood == []:
                return solution,max_value,state_size(solution,values)                
            aux = random_state(neighborhood)
            aux_value = evaluate_state(aux,items)
            aux_size = state_size(aux,items)
            state_value = evaluate_state(state,items)
            if aux_value > state_value and aux_size <= max_size:
                state = aux
                if aux_value > max_value:
                    solution = aux
                    max_value = aux_value
            else:
                if aux_size <= max_size:
                    if change_probability(aux_value,state_value,t):
                        state = aux
        t = t*alfa
        end = time.process_time()

    return solution, state_size(solution,items), max_value


https://pt.wikipedia.org/wiki/Simulated_annealing

https://medium.com/swlh/how-to-implement-simulated-annealing-algorithm-in-python-ab196c2f56a0

https://machinelearningmastery.com/simulated-annealing-from-scratch-in-python/