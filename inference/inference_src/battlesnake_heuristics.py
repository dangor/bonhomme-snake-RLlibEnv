# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# or in the "license" file accompanying this file. This file is distributed 
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
# express or implied. See the License for the specific language governing 
# permissions and limitations under the License.

import numpy as np
import random
try:
    from heuristics import Heuristics
except ModuleNotFoundError:
    from inference.inference_src.heuristics import Heuristics

class MyBattlesnakeHeuristics(Heuristics):
    '''
    The BattlesnakeHeuristics class allows you to define handcrafted rules of the snake.
    '''
    FOOD_INDEX = 0
    def __init__(self):
        pass
       
    @Heuristics.negative_heuristics
    def banned_wall_hits(self, state, snake_id, turn_count, health, json):
        '''
        Heuristics to stop the snakes from hitting a wall.
        '''
        your_snake_body = json["you"]["body"]
        i, j = your_snake_body[0]["y"], your_snake_body[0]["x"]
        
        if -1 in state:
            border_len = int((state.shape[0] - json["board"]["height"])/2)
            i, j = i + border_len, j + border_len
        
        up = state[i-1, j, 1] != -1
        down = state[i+1, j, 1] != -1
        left = state[i, j-1, 1] != -1
        right = state[i, j+1, 1] != -1
        
        action = [up, down, left, right]
                
        return action
    
    @Heuristics.negative_heuristics
    def banned_forbidden_moves(self, state, snake_id, turn_count, health, json):
        '''
        Heuristics to stop the snakes from forbidden moves.
        '''
        your_snake_body = json["you"]["body"]
        i, j = your_snake_body[0]["y"], your_snake_body[0]["x"]
        if len(your_snake_body) == 1:
            return [True, True, True, True]
        next_i, next_j = your_snake_body[1]["y"], your_snake_body[1]["x"]
        
        if -1 in state:
            border_len = int((state.shape[0] - json["board"]["height"])/2)
            i, j = i + border_len, j + border_len
            next_i, next_j = next_i + border_len, next_j + border_len

        up = not (i-1 == next_i and j == next_j)
        down = not (i+1 == next_i and j == next_j)
        left = not (i == next_i and j-1 == next_j)
        right = not (i == next_i and j+1 == next_j)
      
        return [up, down, left, right]

    class Coord:
        def __init__(self, coord):
            self.coord = coord
        
        def isAbove(self, other):
            return self.coord["x"] == other["x"] and self.coord["y"] == other["y"]-1
        def isBelow(self, other):
            return self.coord["x"] == other["x"] and self.coord["y"] == other["y"]+1
        def isLeftOf(self, other):
            return self.coord["x"] == other["x"]-1 and self.coord["y"] == other["y"]
        def isRightOf(self, other):
            return self.coord["x"] == other["x"]+1 and self.coord["y"] == other["y"]
        def isOneAwayFrom(self, other):
            return (self.coord["x"] == other["x"] and abs(self.coord["y"] - other["y"]) == 1 or
                    abs(self.coord["x"] - other["x"]) == 1 and self.coord["y"] == other["y"])
    
    # @Heuristics.negative_heuristics
    def banned_snake_bodies(self, state, snake_id, turn_count, health, json):
        '''
        Heuristics to stop the snakes from hitting other snakes.
        '''
        your_head = self.Coord(json["you"]["body"][0])
        
        up, down, left, right = True, True, True, True
        
        for snake in json["board"]["snakes"]:
            for other_body in snake["body"][0:-1]:
                if your_head.isAbove(other_body):
                    down = False
                elif your_head.isBelow(other_body):
                    up = False
                elif your_head.isLeftOf(other_body):
                    right = False
                elif your_head.isRightOf(other_body):
                    left = False
        
        return [up, down, left, right]

    @Heuristics.negative_heuristics
    def avoid_head_on(self, state, snake_id, turn_count, health, json):
        '''
        Heuristics to avoid head-on collisions.
        '''
        your_head = json["you"]["body"][0]
        x, y = your_head["x"], your_head["y"]

        up, down, left, right = True, True, True, True

        for snake in json["board"]["snakes"]:
            if snake["id"] != snake_id and len(snake["body"]) > 0:
                other_head = self.Coord(snake["body"][0])
                if other_head.isOneAwayFrom({"x": x, "y": y-1}):
                    up = False
                elif other_head.isOneAwayFrom({"x": x, "y": y+1}):
                    down = False
                elif other_head.isOneAwayFrom({"x": x-1, "y": y}):
                    left = False
                elif other_head.isOneAwayFrom({"x": x+1, "y": y}):
                    right = False

        return [up, down, left, right]
    
    @Heuristics.positive_heuristics
    def go_to_food_if_close(self, state, snake_id, turn_count, health, json):
        '''
        Example heuristic to move towards food if it's close to you.
        '''
        if health[snake_id] > 30:
            return [True, True, True, True]  
        
        # Get the position of the snake head
        your_snake_body = json["you"]["body"]
        i, j = your_snake_body[0]["y"], your_snake_body[0]["x"]
        
        # Set food_direction towards food
        food = state[:, :, self.FOOD_INDEX]
        
        # Note that there is a -1 border around state so i = i + 1, j = j + 1
        if -1 in state:
            i, j = i+1, j+1
        
        food_direction = None
        if food[i-1, j] == 1:
            return [True, False, False, False]
        if food[i+1, j] == 1:
            return [False, True, False, False]
        if food[i, j-1] == 1:
            return [False, False, True, False]
        if food[i, j+1] == 1:
            return [False, False, False, True]
        
        return [True, True, True, True]   
    
    def run(self, state, snake_id, turn_count, health, json, action):
        '''
        The main function of the heuristics.
        
        Parameters:
        -----------
        `state`: np.array of size (map_size[0]+2, map_size[1]+2, 1+number_of_snakes)
        Provides the current observation of the gym.
        Your target snake is state[:, :, snake_id+1]
    
        `snake_id`: int
        Indicates the id where id \in [0...number_of_snakes]
    
        `turn_count`: int
        Indicates the number of elapsed turns
    
        `health`: dict
        Indicates the health of all snakes in the form of {int: snake_id: int:health}
        
        `json`: dict
        Provides the same information as above, in the same format as the battlesnake engine.

        `action`: np.array of size 4
        The qvalues of the actions calculated. The 4 values correspond to [up, down, left, right]
        '''
        log_string = ""
        
        wall_masks = self.banned_wall_hits(state, snake_id, turn_count, health, json)
        forbidden_move_masks = self.banned_forbidden_moves(state, snake_id, turn_count, health, json)
        snake_bodies_masks = self.banned_snake_bodies(state, snake_id, turn_count, health, json)
        
        banned_move_mask = np.logical_not(np.array(wall_masks) * np.array(forbidden_move_masks) * np.array(snake_bodies_masks)) * -1e6
        updated_action = action + np.array(banned_move_mask)
        
        head_on_mask = self.avoid_head_on(state, snake_id, turn_count, health, json)
        updated_action = updated_action + (np.logical_not(np.array(head_on_mask)) * -1e4)
        
        go_to_food_mask = self.go_to_food_if_close(state, snake_id, turn_count, health, json)
        updated_action = updated_action + (np.array(go_to_food_mask) * 1e3)

        best_action = int(np.argmax(updated_action))

        if best_action not in [0, 1, 2, 3]:
            log_string += "Random "
            best_action = random.choice([0, 1, 2, 3])
        return best_action, log_string