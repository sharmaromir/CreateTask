import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as conspire 


pygame.init()
pygame.font.init()
font = pygame.font.Font(None, 36) #create the font
header = pygame.font.Font(None, 54) #header font
screen = pygame.display.set_mode((1280, 720))
conspire.switch_backend('Agg')

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def load(self, name):
        self.model.load_state_dict(torch.load(name))





class Game:
    def __init__(self, game_width, game_height, highscore): 
        self.game_width = game_width
        self.game_height = game_height
        self.record = highscore
        self.gameDisplay = pygame.display.set_mode((game_width, game_height))
        self.crash = False
        self.wall_size = 90
        self.gravity = 2400
        self.jump_velocity = 750
        self.wall_speed = (np.power(10, 1/3)+6)*60
        self.score = 0
        self.player = Player(self)
        self.walls = [Wall(self, 0), Wall(self, 1), Wall(self, 2)]
        self.clock = pygame.time.Clock()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def update_score(self): #done
        if(self.walls[0].x <= -self.wall_size): #if a wall reaches the left side of the screen, you cleared it
            self.score += 1
            self.walls[0].wall_coord()
            holder = [self.walls[1], self.walls[2], self.walls[0]]
            self.walls[0] = holder[0]
            self.walls[1] = holder[1]
            self.walls[2] = holder[2]
            return True
        return False
    
    def reset(self):
        # pygame.quit()
        # pygame.init()
        pygame.event.clear()
        self.gameDisplay = pygame.display.set_mode((self.game_width, self.game_height))
        self.crash = False
        self.wall_size = 90
        self.gravity = 2400
        self.jump_velocity = 750
        self.wall_speed = (np.power(10, 1/3)+6)*60
        self.score = 0
        self.player = Player(self)
        self.walls = [Wall(self, 0), Wall(self, 1), Wall(self, 2)]
        self.clock = pygame.time.Clock()
        return self.get_state()
    
    def get_state(self):
        return [self.player.speed, self.player.y, self.walls[0].x, self.walls[0].max_y, self.walls[0].min_y]
        

    def display_ui(self): #done
        if(self.score > self.record):
            self.record = self.score
        myfont = pygame.font.SysFont('Segoe UI', 20)
        myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
        text_score = myfont.render('SCORE: ', True, 'white')
        text_score_number = myfont.render(str(self.score), True, 'white')
        text_highest = myfont.render('HIGHEST SCORE: ', True, 'white')
        text_highest_number = myfont_bold.render(str(self.record), True, 'white')
        self.gameDisplay.blit(text_score, (20, 20))
        self.gameDisplay.blit(text_score_number, (80, 20))
        self.gameDisplay.blit(text_highest, (20, 45))
        self.gameDisplay.blit(text_highest_number, (160, 45))

    def step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        iters = 1
        reward = 1
        if(action==1):
            self.player.speed = -self.jump_velocity
        
        for _ in range(iters):
            self.player.y += self.player.speed/60
            if(self.player.y <= 0 and self.player.speed<0):
                self.player.speed = -self.player.speed
            self.player.speed += self.gravity/60

            for i in range(3):
                self.walls[i].x-=self.wall_speed/60
                player_pos = pygame.Vector2(self.player.x, self.player.y)
                y_dist = min(player_pos.y - self.walls[i].min_y, self.walls[i].max_y-player_pos.y)
                if(y_dist <= self.player.size): 
                    if(y_dist <= 0): #you're above/below the gap, check if you're within PLAYER_SIZE of the wall in the x-direction
                        if abs(player_pos.x-(self.walls[i].x+self.wall_size/2)) <= self.wall_size/2+self.player.size:
                            self.crash = True
                    else:
                        if(abs(player_pos.x-(self.walls[i].x+self.wall_size/2)) <= self.wall_size/2): 
                        #you're in between the two walls, and your y-pos collides
                            self.crash = True
                        else: #check the distance to the four corners
                            topleft = player_pos.distance_to(pygame.Vector2(self.walls[i].x, self.walls[i].min_y))
                            topright = player_pos.distance_to(pygame.Vector2(self.walls[i].x, self.walls[i].max_y))
                            bottomleft = player_pos.distance_to(pygame.Vector2(self.walls[i].x+self.wall_size, self.walls[i].min_y))
                            bottomright = player_pos.distance_to(pygame.Vector2(self.walls[i].x+self.wall_size, self.walls[i].max_y))
                            if min(topleft, topright, bottomleft, bottomright) <= self.player.size:
                                self.crash = True     
            if self.player.y >= self.game_height+self.player.size:
                self.crash = True
            
            self.update_score()

            self.player.update_position(self.player.x, self.player.y)
            if display:
                self.render()
            if(self.crash):
                break
        if not self.crash:
            reward = -max(self.player.y-self.walls[0].max_y, self.walls[0].min_y-self.player.y)
            # scale = 0.3
            # if(self.player.y < self.walls[0].min_y-scale*self.walls[0].x and self.player.y > self.walls[0].max_y+scale*self.walls[0].x):
            #     reward += 50
        
        return self.get_state(), reward, self.crash, {}

    def render(self): #done
        self.gameDisplay.fill('black')
        self.display_ui()
        self.player.display_player()
        for i in range(3):
            self.walls[i].display_wall()
        self.update_screen()

    def update_screen(self):
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(60)

class Player(object): #done
    def __init__(self, gameClone):
        self.game = gameClone
        self.size = 10
        self.x = self.size + self.game.wall_size
        self.y = self.game.game_height * 0.5
        self.speed = 0

    def update_position(self, x, y):
        self.x = x
        self.y = y

    def display_player(self):
        if self.game.crash == False:
            pygame.draw.circle(self.game.gameDisplay, 'red', pygame.Vector2(self.x, self.y), self.size)

class Wall(object): #done
    def __init__(self, gameClone, iter):
        self.game = gameClone
        self.color = "hot pink"
        self.x = self.game.game_width*(1+iter/3)+(iter*self.game.wall_size)/3
        center = random.randint((int)(self.game.game_height/4), (int)(self.game.game_height*3/4)) #choose a random center
        radius = random.randint((int)(self.game.game_height/6), (int)(self.game.game_height/4)) #choose a random width
        self.min_y, self.max_y = max(center - radius, 0), min(center + radius, self.game.game_height)

    def wall_coord(self):
        self.x = self.game.game_width
        center = random.randint((int)(self.game.game_height/4), (int)(self.game.game_height*3/4)) #choose a random center
        radius = random.randint((int)(self.game.game_height/6), (int)(self.game.game_height/4)) #choose a random width
        self.min_y, self.max_y = max(center - radius, 0), min(center + radius, self.game.game_height)

    def display_wall(self):
        pygame.draw.rect(self.game.gameDisplay, self.color, (self.x, 0, self.game.wall_size, self.min_y))
        pygame.draw.rect(self.game.gameDisplay, self.color, (self.x, self.max_y, self.game.wall_size, self.game.game_height-self.max_y))

def devious_scheme(y1): #plot
    y2 = np.convolve(y1, np.ones(100)/100, 'valid')
    #conspire.plot(np.arange(len(y1)), y1, label = "Raw Score")
    conspire.plot(np.arange(len(y2)), y2, label = f"{np.max(y1)}")

    conspire.xlabel('Episode')

    conspire.ylabel('Score')
    
    conspire.legend()
    conspire.savefig('plot.png')
    conspire.close()

if __name__ == "__main__":
    r = open("highscore.in", "r") #reading highscores
    highscore = highscore = (int)(r.read())
    env = Game(1280, 720, highscore)
    state_size = 5
    action_size = 2
    agent = DQNAgent(state_size, action_size)
    agent.load("./jumpy-dqn.pth")
    global display 
    display = True

    for episode in range(10):
        state = env.reset()
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
        print(f"Run {episode}/10 scored: {env.score} points" )
            
