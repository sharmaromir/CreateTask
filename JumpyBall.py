import pygame #drawing
import random #random number generation
import math
import pickle #reinforcement learning model and replays

# defining constants
NUM_WALLS = 3 #constant, the number of walls you can collide with
WALL_START_SPEED = 3 #constant, the starting speed for a wall
PLAYER_SIZE = 10 #constant, radius of the player
WALL_SIZE = 30 #constant, thickness of player
FONT_SIZE = 36 #text size
HEADER_SIZE = 54 #header size
DIFFICULTY_COLORS = ["green", "blue", "purple"] #colors for easy, medium, hard
DIFFICULTY_MAP = ["Easy", "Medium", "Hard"] #names of difficulties
FRAMERATE = 25 #max frames per second
INPUT_DELAY = FRAMERATE/6 #minimum frames between jumps
PLAYER_COLOR = (255,83,73)

#initializing screen
HIGHSCORE_REPLAY = []
with open('replay.pkl', 'rb') as hr:
        HIGHSCORE_REPLAY = pickle.load(hr)
pygame.init()
pygame.font.init()
font = pygame.font.Font(None, FONT_SIZE) #create the font
header = pygame.font.Font(None, 54) #header font
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
r = open("highscore.in", "r") #reading highscores

def create_wall(index) -> tuple[int, int]: #returns two integers representing the bottom and top of the gap
    center = random.randint(screen.get_height()*1/4, screen.get_height()*3/4) #choose a random center
    radius = random.randint(screen.get_height()/6, screen.get_height()/4) #choose a random width
    return max(center - radius, 0), min(center + radius, screen.get_height())

def menu(gravity, jump_velocity, input_time, idle_time, speed, score, highscore, difficulty, highscore_replay) -> tuple[int, bool]: #returns an integer for difficulty, and a boolean for playing or not
    font = pygame.font.Font(None, 36)

    # Menu options
    difficulties = ['Easy', 'Medium', 'Hard']
    selected_difficulty = difficulties[difficulty]
    difficulty_scalar = difficulty
    options = difficulties + ['Replay Highscore', 'Play', 'Quit', 'Selected Difficulty: ' + selected_difficulty, 'Rules', 'Highscore: ' + str(highscore)]
    option_rects = []
    

    for index, option in enumerate(options):
        text_surface = font.render(option, True, (255, 255, 255))
        if(index <= 5):
           rect = text_surface.get_rect(center=(screen.get_width()//2, 200 + index * 50)) #position each button by the center
        elif(index == 6):
            rect = text_surface.get_rect(bottomleft=(20, screen.get_height()-20)) #position the difficulty indicator by the bottom left
            difficulty_position = rect
        elif(index == 7):
            rect = text_surface.get_rect(bottomright=(screen.get_width()-20, screen.get_height()-20)) 
            #position the rules by the bottom right
        else:
            rect = text_surface.get_rect(topleft=(20,20))
            highscore_position = rect
        option_rects.append((option, rect))

    



    running = True
    while running:
        screen.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT: #if the user closes the tab, quit
                pygame.quit()
            elif event.type == pygame.MOUSEBUTTONDOWN: 
                mouse_pos = event.pos
                for option, rect in option_rects:
                    if rect.collidepoint(mouse_pos):
                        for i in range(len(difficulties)):
                            if option == difficulties[i]:
                                selected_difficulty = option 
                                difficulty_scalar = i #if you click a difficulty option, update the difficulty
                        if option == 'Play':
                            return True, difficulty_scalar #return true and the new difficulty
                        elif option == 'Quit':
                            return False, None #return false to quit 
                        elif option == 'Rules':
                            rules() #open the rules
                        elif(option == "Replay Highscore" and highscore > 0):
                            if(not play_recording(gravity, jump_velocity, input_time, idle_time, speed, score, highscore, highscore_replay)):
                                return False, None

        # draw the menu options
        option_rects[6] = (('Selected Difficulty: ' + selected_difficulty, difficulty_position))
        option_rects[8] = (('Highscore: ' + str(highscore), highscore_position))
        title = 'Jumpy Ball'
        text_surface = header.render(title, True, (255,83,73))
        rect = text_surface.get_rect(center=(screen.get_width()//2, 100))
        screen.blit(text_surface, rect)
        for option, rect in option_rects:
            color = 'white'
            if(option == "Replay Highscore" and highscore == 0):
                color = 'dimgray'
            if(option == DIFFICULTY_MAP[difficulty_scalar]):
                color = 'violetred1'
            elif(option == 'Play'):
                color = (255,83,73)
            # elif(option == 'Quit'):
            #     color = 'red'
            text_surface = font.render(option, True, color)
            screen.blit(text_surface, rect)

        pygame.display.flip()
        clock.tick(FRAMERATE)

def rules():
    running = True

    instructions = [
        "Choose the difficulty by clicking easy, medium, or hard.",
        "Once in the game, press space to jump, and avoid hitting the bottom of the screen or any colored walls.",
        "You may hit the top of the screen, but it will bounce you in the opposite direction.",
        "Avoiding walls successfully gives you points, 1 per wall in easy mode, 2 per wall in medium,",
        "and 3 per wall in hard. Try your best to get a high score!"
    ]

    while running:
        text_surface = font.render('Close Rules', True, (255, 255, 255))
        rect = text_surface.get_rect()
        rect.bottomright = (screen.get_width()-20, screen.get_height()-20)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if rect.collidepoint(mouse_pos):
                    return #go back to menu       
        screen.fill("black")
        screen.blit(text_surface, rect)
        #display each line of the instructions
        for i, line in enumerate(instructions):
            text_surface = font.render(line, True, "white")
            screen.blit(text_surface, (20, 20 + i * 30))
        pygame.display.flip()
        clock.tick(FRAMERATE) 

def play_recording(gravity, jump_velocity, input_time, idle_time, speed, score, highscore, highscore_replay) -> tuple[bool, int]: 
    difficulty = highscore_replay[3][0]
    frame = 0
    next_wall = 0
    next_jump = 0
    start_time = pygame.time.get_ticks()
    #returns a boolean and an integer
    fps_val = str(FRAMERATE) #the value for the FPS
    wall_size = WALL_SIZE*(difficulty+1)
    running = True
    started = False #press space to play
    #gravity and jump_velocity are varied to make the game more playable
    gravity = gravity*((difficulty)/2 + 1)*60
    jump_velocity = jump_velocity*((difficulty)/4 + 1)
    dt = 0
    player_pos = pygame.Vector2(wall_size+PLAYER_SIZE, screen.get_height() / 2) #spawn the player in the center of the screen
    clones = []
    rect_spacing = (screen.get_width() + wall_size)/3 
    rect_pos_x = [screen.get_width(), screen.get_width()+rect_spacing, screen.get_width()+2*rect_spacing] 
    #stagger the rectangles spawning
    min_safe_value = [0, 0, 0]
    max_safe_value = [0, 0, 0]
    for i in range(3):
        min_safe_value[i] = highscore_replay[1][next_wall]
        max_safe_value[i] = highscore_replay[2][next_wall]
        next_wall += 1
    screen.fill("black")

    temp_surface = pygame.Surface(((PLAYER_SIZE*2), (PLAYER_SIZE*2)), pygame.SRCALPHA)
    pygame.draw.circle(temp_surface, PLAYER_COLOR, pygame.Vector2(PLAYER_SIZE, PLAYER_SIZE), PLAYER_SIZE) 
    new_rect = temp_surface.get_rect(center=player_pos)
    screen.blit(temp_surface, new_rect.topleft)

    text_surface = header.render('Press Space to Play!', True, "white")
    text_position = text_surface.get_rect() #center the text
    text_position.center = (screen.get_width() // 2, screen.get_height() // 2)
    screen.blit(text_surface, text_position)

    text = ['Score ' + str(score), "Highscore " + str(highscore), "Selected Difficulty: " + str(DIFFICULTY_MAP[difficulty])]
    rects = []
    for i in range(len(text)):
        text_surface = font.render(text[i], True, (255, 255, 255))
        if(i < 2):
            rects.append(text_surface.get_rect(topleft=(20, 20*i+20)))
        else:
            rects.append(text_surface.get_rect(bottomleft=(20, screen.get_height()-20)))
        screen.blit(text_surface, rects[i])

    pygame.display.flip()
    while not started:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            started = True
            #speed = -500 
        elif keys[pygame.K_ESCAPE]:
            running = False
            started = True
        clock.tick(FRAMERATE)

    
    while running: 
        input_time += 1 #track how long it has been since the last input
        speed += gravity*dt #account for gravity
        wall_speed = (pow(10*score/(difficulty+1)+10, 1/3)+(difficulty+1)*WALL_START_SPEED)*60
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        if highscore_replay[0][next_jump]<=pygame.time.get_ticks() - start_time and (next_jump==0 or highscore_replay[0][next_jump]>highscore_replay[0][next_jump-1]):
            next_jump += 1
            clones.append([100, 0, player_pos.x, player_pos.y, 255])
            speed = -jump_velocity #start moving up when you "jump"
            input_time = 0 #reset time since last input
            
        player_pos.y += speed*dt
        #spawning clones
        # fill the screen with a color to wipe away anything from last frame
        screen.fill("black")

        if(player_pos.y <= 0): #if you would go above the map, bounce down
            player_pos.y = 0
            if(speed < 0):
                speed = -speed
        if len(clones)!=0:
            clones = spawn_clones(clones, dt, math.atan2(wall_speed, jump_velocity*2), frame, replay=[])
        for i in range(len(rect_pos_x)): #update walls
            rect_pos_x[i]-=wall_speed*dt
            if(rect_pos_x[i] <= -wall_size): #if a wall reaches the left side of the screen, you cleared it
                rect_pos_x[i] = screen.get_width()
                score += difficulty+1
                gravity += 60 #allow players to move faster as the game progresses
                jump_velocity += 10
                min_safe_value[i] = highscore_replay[1][next_wall]
                max_safe_value[i] = highscore_replay[2][next_wall]
                next_wall += 1
        

        #Draw images on the screen after updating everything    
        temp_surface = pygame.Surface(((PLAYER_SIZE*2), (PLAYER_SIZE*2)), pygame.SRCALPHA)
        pygame.draw.circle(temp_surface, PLAYER_COLOR, pygame.Vector2(PLAYER_SIZE, PLAYER_SIZE), PLAYER_SIZE) 
        new_rect = temp_surface.get_rect(center=player_pos)
        blit(temp_surface, new_rect.topleft, frame, replay=[])
        for i in range(len(rect_pos_x)):
            temp_surface = pygame.Surface((wall_size, screen.get_height()), pygame.SRCALPHA)
            pygame.draw.rect(temp_surface, DIFFICULTY_COLORS[difficulty], (0, 0, wall_size, min_safe_value[i]))
            pygame.draw.rect(temp_surface, DIFFICULTY_COLORS[difficulty], (0, max_safe_value[i], wall_size, screen.get_height() - max_safe_value[i]))
            new_rect = temp_surface.get_rect(topleft=(rect_pos_x[i],0))
            blit(temp_surface, new_rect.topleft, frame, replay=[])
        # top coordinate pair x,y followed by width then height
        text = ['Score ' + str(score), "Highscore " + str(highscore), "Selected Difficulty: " + str(DIFFICULTY_MAP[difficulty])]
        rects = []
        for i in range(len(text)):
            text_surface = font.render(text[i], True, (255, 255, 255))
            if(i < 2):
                rects.append(text_surface.get_rect(topleft=(20, 20*i+20)))
            else:
                rects.append(text_surface.get_rect(bottomleft=(20, screen.get_height()-20)))
            blit(text_surface, rects[i], frame, replay=[])

        #Check if you've lost the game
        x_hit = False
        for i in range(len(rect_pos_x)):
            y_dist = min(player_pos.y - min_safe_value[i], max_safe_value[i]-player_pos.y)
            if(y_dist <= PLAYER_SIZE): 
                if(y_dist <= 0): #you're above/below the gap, check if you're within PLAYER_SIZE of the wall in the x-direction
                    if abs(player_pos.x-(rect_pos_x[i]+wall_size/2)) <= wall_size/2+PLAYER_SIZE:
                        x_hit = True
                else:
                    if(abs(player_pos.x-(rect_pos_x[i]+wall_size/2)) <= wall_size/2): 
                    #you're in between the two walls, and your y-pos collides
                        x_hit = True
                    else: #check the distance to the four corners
                        topleft = player_pos.distance_to(pygame.Vector2(rect_pos_x[i], min_safe_value[i]))
                        topright = player_pos.distance_to(pygame.Vector2(rect_pos_x[i], max_safe_value[i]))
                        bottomleft = player_pos.distance_to(pygame.Vector2(rect_pos_x[i]+wall_size, min_safe_value[i]))
                        bottomright = player_pos.distance_to(pygame.Vector2(rect_pos_x[i]+wall_size, max_safe_value[i]))
                        if min(topleft, topright, bottomleft, bottomright) <= PLAYER_SIZE:
                            x_hit = True     
        running = not x_hit
        if player_pos.y >= screen.get_height()+PLAYER_SIZE: #if you fall off the map, end the game
            running = False    
        if(running == False): #if you've lost, display the losing screen
            text_surface = header.render('End of Replay, Press Escape to Exit.', True, "white")
            text_position = text_surface.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2)) #center the text
            screen.blit(text_surface, text_position)

        # limits FPS
        # use dt for framerate independent phyiscs
        dt = clock.tick(FRAMERATE) / 1000 
        if(input_time == 0): #update fps 
            fps_val = str(int(1/dt)) #only update every input for readability
        text_surface = font.render('FPS: ' + fps_val, True, "white")
        text_position = text_surface.get_rect(topright=(screen.get_width()-20, 20)) #place in top right
        blit(text_surface, text_position, frame, replay=[])
        frame += 1
        
        #add a disclaimer that you're watching a replay
        text_surface = font.render('You are Watching a Replay, Press Escape to Exit.', True, "white")
        text_position = text_surface.get_rect(bottomright=(screen.get_width()-20, screen.get_height()-20)) #center the text
        screen.blit(text_surface, text_position)
        #flip() the display to put your work on screen
        pygame.display.flip()
    while(True): #check for inputs until you get one
        idle_time += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        key = pygame.key.get_pressed()
        if(key[pygame.K_ESCAPE]):
            return True

        clock.tick(FRAMERATE)

def spawn_clones(clones, dt, angle_radians, frame, replay):
    angle_degrees = math.degrees(angle_radians)
    speed = 100  # base starting speed

    for index, clone in enumerate(clones):
        disappear_frames = FRAMERATE/2 #how long before the jump particles disappear
        
        v_x = -math.sin(angle_radians) * speed
        v_y = math.cos(angle_radians) * speed #respective velocities in x and y directions

        clone[0] = max(clone[0]-v_y/disappear_frames, 0) #slow down the air particles
        clone[1] = v_x  #represents x velocity
        clone[2] += clone[1]*dt #represents the x position
        clone[3] += clone[0]*dt #represents the y position
        clone[4] = max(0, clone[4]-255/disappear_frames) #make the air less opaque

        for i in range(3):
            size = (PLAYER_SIZE) + PLAYER_SIZE*(2-i)*(1-clone[0]/100)
            spread = PLAYER_SIZE*(1-clone[0]/100)
            if(size<=0): #error management
                size = 1
            crescent_surface = pygame.Surface(((size*2), 2*(size*2+spread*(3-i))), pygame.SRCALPHA)
            pygame.draw.circle(crescent_surface, (0, clone[4], clone[4]/2), pygame.Vector2(size, size*2+spread*(3-i)+size+spread*(3-i)), size)
            pygame.draw.circle(crescent_surface, "black", pygame.Vector2(size, size*2+spread*(3-i)+size-PLAYER_SIZE/2+spread*(3-i)), size) 
            # rotate the crescent
            rotated_crescent = pygame.transform.rotate(crescent_surface, -angle_degrees)
            # the center of the crescent remains invarient over rotations, thus set the center as the same 
            new_rect = rotated_crescent.get_rect(center=(clone[2], clone[3]))
            blit(rotated_crescent, new_rect.topleft, frame, replay)
        if clone[3] >= screen.get_height():
            clones.pop(index)
    return clones

def blit(surface, position, frame, replay):
    screen.blit(surface, position)
    surface_data = pygame.image.tostring(surface, 'RGBA')
    size = surface.get_size()
    #replay[frame].append([surface_data, size, position, 0])

def run_game(gravity, jump_velocity, input_time, idle_time, speed, score, highscore, difficulty, highscore_replay) -> tuple[bool, int, tuple[tuple, tuple, tuple]]: 
    replay = [[],[],[],[]]
    replay[3].append(difficulty)
    start_time = pygame.time.get_ticks()
    frame = 0
    #returns a boolean and an integer
    fps_val = str(FRAMERATE) #the value for the FPS
    wall_size = WALL_SIZE*(difficulty+1)
    running = True
    started = False #press space to play
    #gravity and jump_velocity are varied to make the game more playable
    gravity = gravity*((difficulty)/2 + 1)*60
    jump_velocity = jump_velocity*((difficulty)/4 + 1)
    dt = 0
    player_pos = pygame.Vector2(wall_size+PLAYER_SIZE, screen.get_height() / 2) #spawn the player in the center of the screen
    clones = []
    rect_spacing = (screen.get_width() + wall_size)/3 
    rect_pos_x = [screen.get_width(), screen.get_width()+rect_spacing, screen.get_width()+2*rect_spacing] 
    #stagger the rectangles spawning
    min_safe_value = [0, 0, 0]
    max_safe_value = [0, 0, 0]
    for i in range(3):
        min_safe_value[i], max_safe_value[i] = create_wall(i) #randomly generate the walls
        replay[1].append(min_safe_value[i])
        replay[2].append(max_safe_value[i])
    screen.fill("black")

    temp_surface = pygame.Surface(((PLAYER_SIZE*2), (PLAYER_SIZE*2)), pygame.SRCALPHA)
    pygame.draw.circle(temp_surface, PLAYER_COLOR, pygame.Vector2(PLAYER_SIZE, PLAYER_SIZE), PLAYER_SIZE) 
    new_rect = temp_surface.get_rect(center=player_pos)
    screen.blit(temp_surface, new_rect.topleft)

    text_surface = header.render('Press Space to Play!', True, "white")
    text_position = text_surface.get_rect() #center the text
    text_position.center = (screen.get_width() // 2, screen.get_height() // 2)
    screen.blit(text_surface, text_position)

    text = ['Score ' + str(score), "Highscore " + str(highscore), "Selected Difficulty: " + str(DIFFICULTY_MAP[difficulty])]
    rects = []
    for i in range(len(text)):
        text_surface = font.render(text[i], True, (255, 255, 255))
        if(i < 2):
            rects.append(text_surface.get_rect(topleft=(20, 20*i+20)))
        else:
            rects.append(text_surface.get_rect(bottomleft=(20, screen.get_height()-20)))
        screen.blit(text_surface, rects[i])

    pygame.display.flip()
    while not started:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            started = True
            #speed = -500 
        elif keys[pygame.K_ESCAPE]:
            running = False
            started = True
        clock.tick(FRAMERATE)

    
    while running: 
        input_time += 1 #track how long it has been since the last input
        speed += gravity*dt #account for gravity
        wall_speed = (pow(10*score/(difficulty+1)+10, 1/3)+(difficulty+1)*WALL_START_SPEED)*60
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] and input_time >= INPUT_DELAY:
            clones.append([100, 0, player_pos.x, player_pos.y, 255])
            speed = -jump_velocity #start moving up when you "jump"
            replay[0].append(pygame.time.get_ticks() - start_time)
            input_time = 0 #reset time since last input
            
        player_pos.y += speed*dt
        #spawning clones
        # fill the screen with a color to wipe away anything from last frame
        screen.fill("black")

        if(player_pos.y <= 0): #if you would go above the map, bounce down
            player_pos.y = 0
            if(speed < 0):
                speed = -speed
        if len(clones)!=0:
            clones = spawn_clones(clones, dt, math.atan2(wall_speed, jump_velocity*2), frame, replay)
        for i in range(len(rect_pos_x)): #update walls
            rect_pos_x[i]-=wall_speed*dt
            if(rect_pos_x[i] <= -wall_size): #if a wall reaches the left side of the screen, you cleared it
                rect_pos_x[i] = screen.get_width()
                score += difficulty+1
                gravity += 60 #allow players to move faster as the game progresses
                jump_velocity += 10
                min_safe_value[i], max_safe_value[i] = create_wall(i)
                replay[1].append(min_safe_value[i])
                replay[2].append(max_safe_value[i])
        

        #Draw images on the screen after updating everything    
        temp_surface = pygame.Surface(((PLAYER_SIZE*2), (PLAYER_SIZE*2)), pygame.SRCALPHA)
        pygame.draw.circle(temp_surface, PLAYER_COLOR, pygame.Vector2(PLAYER_SIZE, PLAYER_SIZE), PLAYER_SIZE) 
        new_rect = temp_surface.get_rect(center=player_pos)
        blit(temp_surface, new_rect.topleft, frame, replay)
        for i in range(len(rect_pos_x)):
            temp_surface = pygame.Surface((wall_size, screen.get_height()), pygame.SRCALPHA)
            pygame.draw.rect(temp_surface, DIFFICULTY_COLORS[difficulty], (0, 0, wall_size, min_safe_value[i]))
            pygame.draw.rect(temp_surface, DIFFICULTY_COLORS[difficulty], (0, max_safe_value[i], wall_size, screen.get_height() - max_safe_value[i]))
            new_rect = temp_surface.get_rect(topleft=(rect_pos_x[i],0))
            blit(temp_surface, new_rect.topleft, frame, replay)
        # top coordinate pair x,y followed by width then height
        text = ['Score ' + str(score), "Highscore " + str(highscore), "Selected Difficulty: " + str(DIFFICULTY_MAP[difficulty])]
        rects = []
        for i in range(len(text)):
            text_surface = font.render(text[i], True, (255, 255, 255))
            if(i < 2):
                rects.append(text_surface.get_rect(topleft=(20, 20*i+20)))
            else:
                rects.append(text_surface.get_rect(bottomleft=(20, screen.get_height()-20)))
            blit(text_surface, rects[i], frame, replay)

        if score >= highscore and score != 0: #if you set a new high score, indicate it
                highscore = score
                for i in range(3):
                    replay[0].append(0)
                    replay[1].append(0)
                    replay[2].append(0)
                highscore_replay = replay
                text_surface = font.render('new best!', True, (255,0,128))
                text_position = (20, 60)
                blit(text_surface, text_position, frame, replay)
                w = open("highscore.in", "w")
                w.write(str(score))

        #Check if you've lost the game
        x_hit = False
        for i in range(len(rect_pos_x)):
            y_dist = min(player_pos.y - min_safe_value[i], max_safe_value[i]-player_pos.y)
            if(y_dist <= PLAYER_SIZE): 
                if(y_dist <= 0): #you're above/below the gap, check if you're within PLAYER_SIZE of the wall in the x-direction
                    if abs(player_pos.x-(rect_pos_x[i]+wall_size/2)) <= wall_size/2+PLAYER_SIZE:
                        x_hit = True
                else:
                    if(abs(player_pos.x-(rect_pos_x[i]+wall_size/2)) <= wall_size/2): 
                    #you're in between the two walls, and your y-pos collides
                        x_hit = True
                    else: #check the distance to the four corners
                        topleft = player_pos.distance_to(pygame.Vector2(rect_pos_x[i], min_safe_value[i]))
                        topright = player_pos.distance_to(pygame.Vector2(rect_pos_x[i], max_safe_value[i]))
                        bottomleft = player_pos.distance_to(pygame.Vector2(rect_pos_x[i]+wall_size, min_safe_value[i]))
                        bottomright = player_pos.distance_to(pygame.Vector2(rect_pos_x[i]+wall_size, max_safe_value[i]))
                        if min(topleft, topright, bottomleft, bottomright) <= PLAYER_SIZE:
                            x_hit = True     
        running = not x_hit
        if player_pos.y >= screen.get_height()+PLAYER_SIZE: #if you fall off the map, end the game
            running = False    
        if(running == False): #if you've lost, display the losing screen
            text_surface = header.render('Game Over! Press Space to Play Again, or Escape to Exit.', True, "white")
            text_position = text_surface.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2)) #center the text
            screen.blit(text_surface, text_position)

        # limits FPS
        # use dt for framerate independent phyiscs
        dt = clock.tick(FRAMERATE) / 1000 
        if(input_time == 0): #update fps 
            fps_val = str(int(1/dt)) #only update every input for readability
        text_surface = font.render('FPS: ' + fps_val, True, "white")
        text_position = text_surface.get_rect(topright=(screen.get_width()-20, 20)) #place in top right
        blit(text_surface, text_position, frame, replay)
        frame += 1
        

        #flip() the display to put your work on screen
        pygame.display.flip()
    while(True): #check for inputs until you get one
        idle_time += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, highscore, highscore_replay
        key = pygame.key.get_pressed()
        if(key[pygame.K_ESCAPE]):
            return False, highscore, highscore_replay
        elif(key[pygame.K_SPACE] and idle_time > 30):
            return True, highscore, highscore_replay
        clock.tick(FRAMERATE)

def main():
    #initializing scores
    highscore_replay = HIGHSCORE_REPLAY
    #game variable setup
    gravity = 20 #how fast the player falls
    jump_velocity = 500 #what speed the player is set to upon jumping
    input_time = 30 #set input to 30 to begin with to ensure players can play
    idle_time = 0 #used for a buffer
    speed = 0 #initialize the player speed to 0
    score = 0 #keep track of what the score is
    highscore = (int)(r.read()) #keep track of high score
    playing = True
    difficulty = 1
    playing, difficulty = menu(gravity, jump_velocity, input_time, idle_time, speed, score, highscore, difficulty, highscore_replay)

    while playing: #run until the player tells you to stop
        playing_game = True
        while playing_game: 
            playing_game, highscore, highscore_replay = run_game(gravity, jump_velocity, input_time, idle_time, speed, score, highscore, difficulty, highscore_replay)
            pygame.display.flip() #update the display
            clock.tick(5) #make sure the previous space press doesn't completely start a new game
        playing, difficulty = menu(gravity, jump_velocity, input_time, idle_time, speed, score, highscore, difficulty, highscore_replay) #return to menu
    pygame.quit()
    with open('replay.pkl', 'wb') as hw:
        pickle.dump(highscore_replay, hw)
        print('overwrote')

if __name__=="__main__": 
    main() 
