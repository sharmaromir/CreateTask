import random
import numpy as np
import pandas as pd
from operator import add
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'

class DQNAgent(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = params['learning_rate']        
        self.epsilon = 1
        self.actual = []
        self.first_layer = params['first_layer_size']
        self.second_layer = params['second_layer_size']
        self.third_layer = params['third_layer_size']
        self.memory = collections.deque(maxlen=params['memory_size'])
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']
        self.optimizer = None
        self.network()
          
    def network(self):
        # Layers
        self.f1 = nn.Linear(18, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, self.third_layer)
        self.f4 = nn.Linear(self.third_layer, 2)
        # weights
        if self.load_weights:
            self.model = self.load_state_dict(torch.load(self.weights))
            print("weights loaded")

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.softmax(self.f4(x), dim=-1)
        return x
    
    def get_state(self, game, player):
        """
        Return the state.
        The state is a numpy array of 18 values, representing:
            - player.y_position 
            - player.x_position
            - player.speed
            - player.size
            - wall size
            - closest wall
            - minimum safe height
            - maximum safe height
            - second closest wall
            - minimum safe height
            - maximum safe height
            - third closest wall
            - minimum safe height
            - maximum safe height
            - wall speed
            - game.gravity
            - game.jump_height
            - score
        """
        state = [
            player.y, 
            player.x,
            player.speed,
            player.size,
            game.wall_size,
            game.wall_speed,
            game.walls[0].x,
            game.walls[0].min_y,
            game.walls[0].max_y,
            game.walls[1].x,
            game.walls[1].min_y,
            game.walls[1].max_y,
            game.walls[2].x,
            game.walls[2].min_y,
            game.walls[2].max_y,
            game.gravity,
            game.jump_velocity,
            game.score
        ]

        return np.asarray(state)

    def set_reward(self, game, crash, player):
        """
        Return the reward.
        The reward is:
            -100 when you crash
            +1 when your score increases
            0 otherwise
        """
        closest = -1
        third = -1
        for i in range(3):
            if(game.walls[i].x <= game.wall_size + 2*player.size):
                third = i
        if third == -1:
            if(game.walls[0].x < game.walls[1].x):
                if(game.walls[0].x < game.walls[2].x):
                    closest = 0
                else:
                    closest = 2
            else:
                if(game.walls[1].x < game.walls[2].x):
                    closest = 1
                else:
                    closest = 2
        else:
            closest = third+1


        self.reward = 0
        if crash:
            self.reward = -100
            return self.reward
        if game.walls[0].x <= -game.wall_size or game.walls[1].x <= -game.wall_size or game.walls[2].x <= -game.wall_size:
            self.reward = 1
        if player.y < game.walls[closest%3].max_y-player.size and player.y > game.walls[closest%3].min_y+player.size:
            self.reward += 0.5
        # if player.y < game.walls[(closest+1)%3].max_y-player.size and player.y > game.walls[(closest+1)%3].min_y+player.size:
        #     self.reward += 0.4
        # if player.y < game.walls[(closest+2)%3].max_y-player.size and player.y > game.walls[(closest+2)%3].min_y+player.size:
        #     self.reward += 0.1
        return self.reward

    def remember(self, state, action, reward, next_state, done):
        """
        Store the <state, action, reward, next_state, is_done> tuple in a 
        memory buffer for replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory, batch_size):
        """
        Replay memory.
        """
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            self.train()
            torch.set_grad_enabled(True)
            target = reward
            next_state_tensor = torch.tensor(np.expand_dims(next_state, 0), dtype=torch.float32).to(DEVICE)
            state_tensor = torch.tensor(np.expand_dims(state, 0), dtype=torch.float32, requires_grad=True).to(DEVICE)
            if not done:
                target = reward + self.gamma * torch.max(self.forward(next_state_tensor)[0])
            output = self.forward(state_tensor)
            target_f = output.clone()
            target_f[0][np.argmax(action)] = target
            target_f.detach()
            self.optimizer.zero_grad()
            loss = F.mse_loss(output, target_f)
            loss.backward()
            self.optimizer.step()            

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Train the DQN agent on the <state, action, reward, next_state, is_done>
        tuple at the current timestep.
        """
        self.train()
        torch.set_grad_enabled(True)
        target = reward
        next_state_tensor = torch.tensor(next_state.reshape((1, 18)), dtype=torch.float32).to(DEVICE)
        state_tensor = torch.tensor(state.reshape((1, 18)), dtype=torch.float32, requires_grad=True).to(DEVICE)
        if not done:
            target = reward + self.gamma * torch.max(self.forward(next_state_tensor[0]))
        output = self.forward(state_tensor)
        target_f = output.clone()
        target_f[0][np.argmax(action)] = target
        target_f.detach()
        self.optimizer.zero_grad()
        loss = F.mse_loss(output, target_f)
        loss.backward()
        self.optimizer.step()