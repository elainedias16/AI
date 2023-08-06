import math
import pygame
import os
import random
import time
from sys import exit

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel, wilcoxon


pygame.init()

# Valid values: HUMAN_MODE or AI_MODE
GAME_MODE = "AI_MODE"
# GAME_MODE = "HUMAN_MODE"
# RENDER_GAME = True #With graphic interface
RENDER_GAME = False #Without graphic interface

# Global Constants
global data 
data = []
global max_value_training 

max_value_training = {}
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100


if RENDER_GAME:
    SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUNNING = [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))]
JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))
DUCKING = [pygame.image.load(os.path.join("Assets/Dino", "DinoDuck1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoDuck2.png"))]

SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))]
LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus4.png"))]

BIRD = [pygame.image.load(os.path.join("Assets/Bird", "Bird1.png")),
        pygame.image.load(os.path.join("Assets/Bird", "Bird2.png"))]

CLOUD = pygame.image.load(os.path.join("Assets/Other", "Cloud.png"))

BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))


def normalize_data(states):
    normalized_data = []
    for state in states:
        clf = state[-1]
        state = state[:-1]
        normalized_state = state
        normalized_state.append(clf)
        normalized_data.append(normalized_state)
    return normalized_data


def normalize_array(array):
    min_val = min(array)
    max_val = max(array)
    normalized_array = [(val - min_val) / (max_val - min_val) for val in array]
    return normalized_array


class Dinosaur:
    X_POS = 90
    Y_POS = 330
    Y_POS_DUCK = 355
    JUMP_VEL = 17
    JUMP_GRAV = 1.1

    def __init__(self):
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = 0
        self.jump_grav = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    def update(self, userInput):
        if self.dino_duck and not self.dino_jump:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 20:
            self.step_index = 0

        if userInput == "K_UP" and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif userInput == "K_DOWN" and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif userInput == "K_DOWN":
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = True
        elif not (self.dino_jump or userInput == "K_DOWN"):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

    def duck(self):
        self.image = self.duck_img[self.step_index // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_duck:
            self.jump_grav = self.JUMP_GRAV * 4
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel
            self.jump_vel -= self.jump_grav
        if self.dino_rect.y > self.Y_POS + 10:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL
            self.jump_grav = self.JUMP_GRAV
            self.dino_rect.y = self.Y_POS

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))

    def getXY(self):
        return (self.dino_rect.x, self.dino_rect.y)


class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))


class Obstacle():
    def __init__(self, image, type):
        super().__init__()
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()

        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < - self.rect.width:
            obstacles.pop(0)

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)

    def getXY(self):
        return (self.rect.x, self.rect.y)

    def getHeight(self):
        return y_pos_bg - self.rect.y

    def getType(self):
        return (self.type)


class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 345


class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 325


class Bird(Obstacle):
    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)

        # High, middle or ground
        if random.randint(0, 3) == 0:
            self.rect.y = 345
        elif random.randint(0, 2) == 0:
            self.rect.y = 260
        else:
            self.rect.y = 300
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 19:
            self.index = 0
        SCREEN.blit(self.image[self.index // 10], self.rect)
        self.index += 1


#My Classifier : KNN
class KeyClassifier:
    def __init__(self, states, k):
        self.states = states
        self.k = k
    
    def euclidian_distance(self, P1, P2):
        distance = 0
        for i in range(len(P1)):
            distance += (P1[i] - P2[i]) ** 2
            
        distance = distance ** (1/2) #sqrt
        return distance
    
    def count_classes(self, items):
        counts = {}
        for item in items:
            if item in counts:
                counts[item] += 1
            else:
                counts[item] = 1
        return counts


    def get_neighbors(self, actual_state):
        distances = []

        for state in self.states:
            state_aux = state[:-1]
            target = state[-1]

            #state_aux is state without target
            distance = self.euclidian_distance(state_aux, actual_state)
            distances.append([distance, target])

        distances.sort(key=lambda x: x[0]) 
       
        #Getting neighbors
        neighbors = []
        for i in range(0, self.k):
            neighbors.append(distances[i])
        return neighbors
        
     

    #KNN
    def keySelector(self, distance, obHeight, speed, obType, nextObDistance, nextObHeight, nextObType):
        actual_state = [distance, speed, obHeight, nextObDistance]
        actual_state = actual_state
        neighbors = self.get_neighbors(actual_state)
       
        classes = [neighbor[1] for neighbor in neighbors]

        count = self.count_classes(classes)
        #In case of a tie, choose the first clf of count dictionary
        clf = max(count, key=count.get)

        if(clf == 1):
            return 'K_UP'
        elif(clf == 0):
            return 'K_NO'
        else:
            return 'K_DOWN'
       

    def updateState(self, state):
        self.state = state
        



def first(x):
    return x[0]


class KeySimplestClassifier(KeyClassifier):
    def __init__(self, state):
        self.state = state

    def keySelector(self, distance, obHeight, speed, obType, nextObDistance, nextObHeight,nextObType):
        self.state = sorted(self.state, key=first)
        for s, d in self.state:
            if speed < s:
                limDist = d
                break
        if distance <= limDist:
            if isinstance(obType, Bird) and obHeight > 50:
                return "K_DOWN"
            else:
                return "K_UP"
        return "K_NO"

    def updateState(self, state):
        self.state = state


def playerKeySelector(distance, obHeight, game_speed, obType, nextObDistance, nextObHeight,nextObType):
    userInputArray = pygame.key.get_pressed()
    if userInputArray[pygame.K_UP]:
        data.append([distance,game_speed, obHeight, nextObDistance,  1])
        return "K_UP"
    elif userInputArray[pygame.K_DOWN]:
        data.append([distance, game_speed,obHeight, nextObDistance, -1])
        return "K_DOWN"
    else:
        data.append([distance, game_speed,obHeight, nextObDistance, 0])
        return "K_NO"
    
    
def playGame():
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles
    run = True

    clock = pygame.time.Clock()
    cloud = Cloud()
    font = pygame.font.Font('freesansbold.ttf', 20)

    player = Dinosaur()
    game_speed = 10
    x_pos_bg = 0
    y_pos_bg = 383
    points = 0

    obstacles = []
    death_count = 0
    spawn_dist = 0

   
    def score():
        global points, game_speed
        points += 0.25
        if points % 100 == 0:
            game_speed += 1

        if RENDER_GAME:
            text = font.render("Points: " + str(int(points)), True, (0, 0, 0))
            textRect = text.get_rect()
            textRect.center = (1000, 40)
            SCREEN.blit(text, textRect)


    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
            x_pos_bg = 0
        x_pos_bg -= game_speed

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                exit()

        if RENDER_GAME:
            SCREEN.fill((255, 255, 255))

        distance = 1500
        nextObDistance = 2000
        obHeight = 0
        nextObHeight = 0
        obType = 2
        nextObType = 2
        if len(obstacles) != 0:
            xy = obstacles[0].getXY()
            distance = xy[0]
            obHeight = obstacles[0].getHeight()
            obType = obstacles[0]

        if len(obstacles) == 2:
            nextxy = obstacles[1].getXY()
            nextObDistance = nextxy[0]
            nextObHeight = obstacles[1].getHeight()
            nextObType = obstacles[1]

        if GAME_MODE == "HUMAN_MODE":
            #userInput = playerKeySelector() 
            userInput = playerKeySelector(distance, obHeight, game_speed, obType, nextObDistance, nextObHeight,
                                             nextObType)


        else:
            userInput = aiPlayer.keySelector(distance, obHeight, game_speed, obType, nextObDistance, nextObHeight,
                                             nextObType)
        

        if len(obstacles) == 0 or obstacles[-1].getXY()[0] < spawn_dist:
            spawn_dist = random.randint(0, 670)
            if random.randint(0, 2) == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS))
            elif random.randint(0, 2) == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS))
            elif random.randint(0, 5) == 5:
                obstacles.append(Bird(BIRD))

        player.update(userInput)

        if RENDER_GAME:
            player.draw(SCREEN)

        for obstacle in list(obstacles):
            obstacle.update()
            if RENDER_GAME:
                obstacle.draw(SCREEN)


        if RENDER_GAME:
            background()
            cloud.draw(SCREEN)

        cloud.update()

        score()

        if RENDER_GAME:
            clock.tick(60)
            pygame.display.update()

        for obstacle in obstacles:
            if player.dino_rect.colliderect(obstacle.rect):
                if RENDER_GAME:
                    pygame.time.delay(2000)
                death_count += 1
                return points
            
       
    
 
# Change State Operator
def change_state(state, position, vs, vd):
    aux = state.copy()
    s, d = state[position]
    ns = s + vs
    nd = d + vd
    if ns < 15 or nd > 1000:
        return []
    return aux[:position] + [(ns, nd)] + aux[position + 1:]


# Neighborhood
def generate_neighborhood(state):
    neighborhood = []
    state_size = len(state)
    for i in range(state_size):
        ds = random.randint(1, 10)
        dd = random.randint(1, 100)
        new_states = [change_state(state, i, ds, 0), change_state(state, i, (-ds), 0), change_state(state, i, 0, dd),
                      change_state(state, i, 0, (-dd))]
        for s in new_states:
            if s != []:
                neighborhood.append(s)

    return neighborhood


# Gradiente Ascent
def gradient_ascent(state, max_time):
    start = time.process_time()
    res, max_value = manyPlaysResults(3)
    better = True
    end = 0


    while better and end - start <= max_time:
        neighborhood = generate_neighborhood(state)
        better = False
        for s in neighborhood:
            aiPlayer = KeySimplestClassifier(s)
            res, value = manyPlaysResults(3)
            if value > max_value:
                state = s
                max_value = value
                better = True
        end = time.process_time()
    return state, max_value



# state = [distance, speed, obHeight, nextObDistance]
# Neighborhood
def generate_neighborhood_sm(states):
    neighborhood = []
    for _ in range(3):
        neighbor = []
        for state in states:
            new_state = [state[0] + random.uniform(0, 1), state[1], state[2] + random.uniform(0, 1), state[3] + random.uniform(0, 1), state[4]] #so tinha o random no state 3
            neighbor.append(new_state)
        neighborhood.append(neighbor)
    return neighborhood


def evaluete_state(states, k):
    value = 0
    aiPlayer = KeyClassifier(states, k)
    aiPlayer.updateState(states)
    res, value = manyPlaysResults(3)
    return value


def simulated_annealing(states, temperature, alpha, max_time , iter_max, k):
    best_states = states
    res, max_value = manyPlaysResults(3)
    start = time.process_time()
    end = 0
    while temperature >= 1 and end-start <= max_time:
        print('Temperatura : ' , temperature)
        for i in range(0, iter_max): 
            neighborhood = generate_neighborhood_sm(states)
            for j in range(0, len(neighborhood)):
                cost_neighbor = evaluete_state(neighborhood[j], k)
                cost_states = evaluete_state(states, k)
                delta = cost_states - cost_neighbor

                if delta > 0:
                    best_states = neighborhood[j]
                    max_value = cost_neighbor
                elif(random.uniform(0,1) < math.exp(-delta / temperature)):
                    best_states = neighborhood[j]
                    max_value = cost_neighbor
        
        max_value_training[temperature] = max_value
        print(max_value) 
        temperature = temperature * alpha
        print('\n') 
        end = time.process_time() 

    return best_states, max_value




def manyPlaysResults(rounds):
    results = []
    for round in range(rounds):
        results += [playGame()]
    npResults = np.asarray(results)
    return (results, npResults.mean() - npResults.std())


def create_dataset(filename):
    with open(filename, 'a') as f:
        for line in data:
            f.write(str(line[0]) + ',' + str(line[1]) + ',' + str(line[2])  + ',' + str(line[3]) + ',' +  str(line[4]) + "\n")


def load_states(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            line = line.strip()
            line = line.split(',')    
            data.append([float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])])
        return data

def dataset_sm(filename, dataset):
    with open(filename, 'a') as f:
        for line in dataset:
            f.write(str(line[0]) + ',' + str(line[1]) + ',' + str(line[2])  + ',' + str(line[3]) + ',' +  str(line[4]) + "\n")


def boxplot(my_scores, teacher_scores):


    data = [my_scores, teacher_scores]
    plt.boxplot(data, labels=['My Scores', 'Teacher Scores'], patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.xlabel('Groups')
    plt.ylabel('Scores')
    plt.title('Boxplot of Scores')
    plt.savefig('Boxplot of Scores.png')
    plt.show()


def tests_wilcoxon_ttest_rel(my_scores, teacher_scores):
    scores = []
    clf = ['my_score', 'teacher_score']

    scores.append(my_scores)
    scores.append(teacher_scores)

    size = len(scores)
    matrix_p_values = np.zeros((size, size)).tolist()

    for i_index, i in enumerate(scores):
        for j_index, j in enumerate(scores):
            if i_index == j_index:
                matrix_p_values[i_index][j_index] = clf[i_index]
                continue
            else:
                statistic, p_value = ttest_rel(i, j)  # Teste Pareado
                matrix_p_values[i_index][j_index] = p_value

                statistic, p_value = wilcoxon(i, j, method='approx')
                matrix_p_values[i_index][j_index] = p_value
    df_p_values = pd.DataFrame(matrix_p_values)
    return df_p_values


def table_results(my_scores, teacher_scores):
    data = {'Coluna1': teacher_scores, 'Coluna2': my_scores}
    df = pd.DataFrame(data)

    df.loc['Média'] = df.mean()
    df.loc['Desvio Padrão'] = df.std()

    return df


#Print res, mean, std, value in a file
def res_data(training, res, mean, std, value):
    path = 'training/training' + str(training)
    if not os.path.exists(path):
        os.makedirs(path) 
    with open(path + '/res.txt', 'a') as f:
        f.write('Results \n')
        f.write("[" + ", ".join(str(element) for element in res) + "]\n")
        f.write('Mean std value \n')
        f.write(str(mean) + " " + str(std) + " " + str(value) + "\n \n \n")


def data_training(max_value_training, training):
    with open('training/training' + str(training) + '/max_value.txt', 'a') as f:
        for temperatura, max_value in max_value_training.items():
            f.write(f'{temperatura}, {max_value}\n')

def main():
    global aiPlayer

    '''Best Parameters found'''
    k = 5
    temperature = 900
    alpha = 0.8
    max_time = 5500
    iter_max = 2


    initial_states_with_targets = load_states('data/test3.txt')
    aiPlayer = KeyClassifier(initial_states_with_targets, k)

    new_states_targets, max_value = simulated_annealing(initial_states_with_targets, temperature, alpha, max_time , iter_max, k)
    aiPlayer = KeyClassifier(new_states_targets, k)
    res, value = manyPlaysResults(30)
    npRes = np.asarray(res)
    print(res, npRes.mean(), npRes.std(), value)
    print()

    training = 23
    res_data(training, res, npRes.mean(), npRes.std(), value)
    data_training(max_value_training, training)
    #Getting the base of simulated annealing
    dataset_sm('training/training' + str(training) + '/states.txt' , new_states_targets)



    # teacher_scores = [1214.0, 759.5, 1164.25, 977.25, 1201.0, 930.0, 1427.75, 799.5, 1006.25, 783.5, 728.5, 419.25, 1389.5, 730.0,
    #                 1306.25, 675.5, 1359.5, 1000.25, 1284.5, 1350.0, 751.0, 1418.75, 1276.5, 1645.75, 860.0, 745.5, 1426.25, 783.5,
    #                 1149.75, 1482.25]
    # boxplot(res, teacher_scores)
    # table_results(res, teacher_scores)
    # # table_results(res, teacher_scores)
    # print( table_results(res, teacher_scores))
    # print('\nTestes wilcoxon_ttest ')
    # tests_wilcoxon_ttest_rel(res, teacher_scores)
    # print(tests_wilcoxon_ttest_rel(res, teacher_scores))
    


main()
