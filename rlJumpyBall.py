import os
import pygame
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from DQN import DQNAgent
from random import randint
import random
import statistics
import torch.optim as optim
import torch 
from GPyOpt.methods import BayesianOptimization
from bayesOpt import *
import datetime
import distutils.util
DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'

#################################
#   Define parameters manually  #
#################################
def define_parameters():
    params = dict()
    # Neural Network
    params['epsilon_decay_linear'] = 1/100
    params['learning_rate'] = 0.00013629
    params['first_layer_size'] = 200    # neurons in the first layer
    params['second_layer_size'] = 20   # neurons in the second layer
    params['third_layer_size'] = 50    # neurons in the third layer
    params['episodes'] = 250          
    params['memory_size'] = 2500
    params['batch_size'] = 1000
    # Settings
    params['weights_path'] = 'weights/weights.h5'
    params['train'] = True
    params["test"] = False
    params['plot_score'] = True
    params['log_path'] = 'logs/scores_' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) +'.txt'
    return params


class Game:
    """ Initialize PyGAME """
    
    def __init__(self, game_width, game_height): #TO DO
        self.game_width = game_width
        self.game_height = game_height
        self.gameDisplay = pygame.display.set_mode((game_width, game_height))
        self.crash = False
        self.wall_size = 90
        self.gravity = 2400
        self.jump_velocity = 750
        self.wall_speed = (np.power(10, 1/3)+6)*60
        self.score = 0
        self.player = Player(self)
        self.walls = [Wall(self, 0), Wall(self, 1), Wall(self, 2)]
        self.clock = pygame.time.Clock


class Player(object): #done
    def __init__(self, game):
        self.size = 10
        self.x = self.size + game.wall_size
        self.y = game.game_height * 0.5
        self.speed = 0

    def update_position(self, x, y):
        self.x = x
        self.y = y

    def do_move(self, move, game, agent):
        iters = 1
        if(np.array_equal(move, [1,0])):
            self.speed = -game.jump_velocity
            iters = 10
        
        for _ in range(iters):
            self.y += self.speed/60
            if(self.y <= 0 and self.speed<0):
                self.speed = -self.speed
            self.speed += game.gravity/60

            for i in range(3):
                game.walls[i].x-=game.wall_speed/60
                player_pos = pygame.Vector2(self.x, self.y)
                y_dist = min(player_pos.y - game.walls[i].min_y, game.walls[i].max_y-player_pos.y)
                if(y_dist <= self.size): 
                    if(y_dist <= 0): #you're above/below the gap, check if you're within PLAYER_SIZE of the wall in the x-direction
                        if abs(player_pos.x-(game.walls[i].x+game.wall_size/2)) <= game.wall_size/2+self.size:
                            game.crash = True
                    else:
                        if(abs(player_pos.x-(game.walls[i].x+game.wall_size/2)) <= game.wall_size/2): 
                        #you're in between the two walls, and your y-pos collides
                            game.crash = True
                        else: #check the distance to the four corners
                            topleft = player_pos.distance_to(pygame.Vector2(game.walls[i].x, game.walls[i].min_y))
                            topright = player_pos.distance_to(pygame.Vector2(game.walls[i].x, game.walls[i].max_y))
                            bottomleft = player_pos.distance_to(pygame.Vector2(game.walls[i].x+game.wall_size, game.walls[i].min_y))
                            bottomright = player_pos.distance_to(pygame.Vector2(game.walls[i].x+game.wall_size, game.walls[i].max_y))
                            if min(topleft, topright, bottomleft, bottomright) <= self.size:
                                game.crash = True     
            if self.y >= game.game_height+self.size:
                game.crash = True
            
            score(game)

            self.update_position(self.x, self.y)

    def display_player(self, game):
        if game.crash == False:
            pygame.draw.circle(game.gameDisplay, 'red', pygame.Vector2(self.x, self.y), self.size)
        else:
            pygame.time.wait(300)


class Wall(object): #done
    def __init__(self, game, iter):
        self.color = "hot pink"
        self.x = game.game_width*(1+iter/3)
        center = random.randint(game.game_height/4, game.game_height*3/4) #choose a random center
        radius = random.randint(game.game_height/6, game.game_height/4) #choose a random width
        self.min_y, self.max_y = max(center - radius, 0), min(center + radius, game.game_height)

    def wall_coord(self, game):
        self.x = game.game_width
        center = random.randint(game.game_height/4, game.game_height*3/4) #choose a random center
        radius = random.randint(game.game_height/6, game.game_height/4) #choose a random width
        self.min_y, self.max_y = max(center - radius, 0), min(center + radius, game.game_height)

    def display_wall(self, game):
        pygame.draw.rect(game.gameDisplay, self.color, (self.x, 0, game.wall_size, self.min_y))
        pygame.draw.rect(game.gameDisplay, self.color, (self.x, self.max_y, game.wall_size, game.game_height-self.max_y))


def score(game): #done
    for i in range(3):
        if(game.walls[i].x <= -game.wall_size): #if a wall reaches the left side of the screen, you cleared it
            game.score += 1
            game.gravity += 60 #allow players to move faster as the game progresses
            game.jump_velocity += 10
            game.wall_speed = (pow(10*game.score/3+10, 1/3)+(3)*3)*60
            game.walls[i].wall_coord(game)


def get_record(score, record): #done
    if score >= record:
        return score
    else:
        return record


def display_ui(game, score, record): #done
    myfont = pygame.font.SysFont('Segoe UI', 20)
    myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
    text_score = myfont.render('SCORE: ', True, 'white')
    text_score_number = myfont.render(str(score), True, 'white')
    text_highest = myfont.render('HIGHEST SCORE: ', True, 'white')
    text_highest_number = myfont_bold.render(str(record), True, 'white')
    game.gameDisplay.blit(text_score, (20, 20))
    game.gameDisplay.blit(text_score_number, (80, 20))
    game.gameDisplay.blit(text_highest, (20, 45))
    game.gameDisplay.blit(text_highest_number, (160, 45))


def display(player, game, record): #done
    game.gameDisplay.fill('black')
    display_ui(game, game.score, record)
    player.display_player(game)
    for i in range(3):
        game.walls[i].display_wall(game)
    update_screen(game)
    
    


def update_screen(game):
    pygame.event.pump()
    pygame.display.update()


def initialize_game(player, game, agent, batch_size): #done
    state_init1 = agent.get_state(game, player)  # [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
    action = [1, 0]
    player.do_move(action, game, agent)
    state_init2 = agent.get_state(game, player)
    reward1 = agent.set_reward(game, game.crash, player)
    agent.remember(state_init1, action, reward1, state_init2, game.crash)
    agent.replay_new(agent.memory, batch_size)


def plot_seaborn(array_counter, array_score, train):
    sns.set(color_codes=True, font_scale=1.5)
    sns.set_style("white")
    plt.figure(figsize=(13,8))
    fit_reg = False if train== False else True        
    ax = sns.regplot(
        np.array([array_counter])[0],
        np.array([array_score])[0],
        #color="#36688D",
        x_jitter=.1,
        scatter_kws={"color": "#36688D"},
        label='Data',
        fit_reg = fit_reg,
        line_kws={"color": "#F49F05"}
    )
    # Plot the average line
    y_mean = [np.mean(array_score)]*len(array_counter)
    ax.plot(array_counter,y_mean, label='Mean', linestyle='--')
    ax.legend(loc='upper right')
    ax.set(xlabel='# games', ylabel='score')
    plt.show()


def get_mean_stdev(array):
    return statistics.mean(array), statistics.stdev(array)    


def test(params):
    params['load_weights'] = True
    params['train'] = False
    params["test"] = False 
    score, mean, stdev = run(params)
    return score, mean, stdev


def run(params):
    """
    Run the DQN algorithm, based on the parameters previously set.   
    """
    pygame.init()
    agent = DQNAgent(params)
    agent = agent.to(DEVICE)
    agent.optimizer = optim.Adam(agent.parameters(), weight_decay=0, lr=params['learning_rate'])
    counter_games = 0
    score_plot = []
    counter_plot = []
    record = 0
    total_score = 0
    while counter_games < params['episodes']:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # Initialize classes
        game = Game(1280, 720)
        player1 = game.player

        # Perform first move
        initialize_game(player1, game, agent, params['batch_size'])
        if params['display']:
            display(player1, game, record)
        
        steps = 0       # steps since the last positive reward
        while (not game.crash) and (steps < 100):
            if not params['train']:
                agent.epsilon = 0.01
            else:
                # agent.epsilon is set to give randomness to actions
                agent.epsilon = 1 - (counter_games * params['epsilon_decay_linear'])

            # get old state
            state_old = agent.get_state(game, player1)

            # perform random actions based on agent.epsilon, or choose the action
            if random.uniform(0, 1) < agent.epsilon:
                final_move = np.eye(2)[randint(0,1)]
            else:
                # predict action based on the old state
                with torch.no_grad():
                    state_old_tensor = torch.tensor(state_old.reshape((1, 18)), dtype=torch.float32).to(DEVICE)
                    prediction = agent(state_old_tensor)
                    final_move = np.eye(2)[np.argmax(prediction.detach().cpu().numpy()[0])]

            # perform new move and get new state
            player1.do_move(final_move, game, agent)
            state_new = agent.get_state(game, player1)

            # set reward for the new state
            reward = agent.set_reward(game, game.crash, player1)
            
            # if food is eaten, steps is set to 0
            if reward > 0:
                steps = 0
                
            if params['train']:
                # train short memory base on the new action and state
                agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)
                # store the new data into a long term memory
                agent.remember(state_old, final_move, reward, state_new, game.crash)

            record = get_record(game.score, record)
            if params['display']:
                display(player1, game, record)
                pygame.time.delay(params['speed'])
            steps+=1
        if params['train']:
            agent.replay_new(agent.memory, params['batch_size'])
        counter_games += 1
        total_score += game.score
        print(f'Game {counter_games}      Score: {game.score}')
        score_plot.append(game.score)
        counter_plot.append(counter_games)
    mean, stdev = get_mean_stdev(score_plot)
    if params['train']:
        model_weights = agent.state_dict()
        torch.save(model_weights, params["weights_path"])
    if params['plot_score']:
        plot_seaborn(counter_plot, score_plot, params['train'])
    return total_score, mean, stdev

if __name__ == '__main__':
    # Set options to activate or deactivate the game view, and its speed
    pygame.font.init()
    parser = argparse.ArgumentParser()
    params = define_parameters()
    parser.add_argument("--display", nargs='?', type=distutils.util.strtobool, default=True)
    parser.add_argument("--speed", nargs='?', type=int, default=50)
    parser.add_argument("--bayesianopt", nargs='?', type=distutils.util.strtobool, default=False)
    args = parser.parse_args()
    print("Args", args)
    print(args.display)
    params['display'] = args.display
    if(args.display == 1):
        params['display'] = True
    params['speed'] = args.speed
    if args.bayesianopt:
        bayesOpt = BayesianOptimization(params)
        bayesOpt.optimize_RL()
    if params['train']:
        print("Training...")
        params['load_weights'] = False   # when training, the network is not pre-trained
        run(params)
    if params['test']:
        print("Testing...")
        params['train'] = False
        params['load_weights'] = True
        run(params)