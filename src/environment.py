import numpy as np
import random as r
#import pygame

from collections import Counter
from itertools import permutations

# Tile location in matrix
TILE_ID = [
    # Outer layer
    (3, 6), (3, 10), (3, 14),
    (7, 16), (11, 18), (15, 16),
    (19, 14), (19, 10), (19, 6),
    (15, 4), (11, 2), (7, 4),

    # Middle layer
    (7, 8), (7, 12),
    (11, 14), (15, 12),
    (15, 8), (11, 6),

    # Middle point
    (11, 10),
]

# Bandit = 0, Brick = 2, Grain = 3, Ore = 4, Lumber = 5, Wool = 6 (Buildable tile = 1)
TILE_TYPE = [0, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6]
TILE_NUMBER = [5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11]   # Dice number

BUILDABLE = 1

RIGHT_DIAGONAL = [
    (1, 5), (1, 9), (1, 13),
    (5, 3), (5, 7), (5, 11), (5, 15),
    (9, 1), (9, 5), (9, 9), (9, 13), (9, 17),
    (13, 3), (13, 7), (13, 11), (13, 15), (13, 19),
    (17, 5), (17, 9), (17, 13), (17, 17),
    (21, 7), (21, 11), (21, 15),
]
LEFT_DIAGONAL = [
    (1, 7), (1, 11), (1, 15),
    (5, 5), (5, 9), (5, 13), (5, 17),
    (9, 3), (9, 7), (9, 11), (9, 15), (9, 19),
    (13, 1), (13, 5), (13, 9), (13, 13), (13, 17),
    (17, 3), (17, 7), (17, 11), (17, 15),
    (21, 5), (21, 9), (21, 13)
]


class Tile:
    def __init__(self, coords, tile_type):
        self.row, self.col = coords
        self.tile_type = tile_type
        self.number = 0

        if tile_type == 0:
            self.has_robber = True
        else:
            self.has_robber = False

    def get_tile_type(self):
        if self.tile_type == 0:
            return "bandit"
        if self.tile_type == 0.2:
            return "brick"
        if self.tile_type == 0.3:
            return "grain"
        if self.tile_type == 0.4:
            return "ore"
        if self.tile_type == 0.5:
            return "lumber"
        if self.tile_type == 0.6:
            return "wool"


class Environment:
    def __init__(self, players, visualize=False, shuffle=False):
        self.visualize = visualize
        self.shuffle = shuffle     # Shuffle tiles
        r.shuffle(players)
        self.players = players

        self.board = []
        self.tiles = []
        self.create_board()

        # Villages 54, Cities 54, Roads 72, Trades 20
        buildable_locations = np.argwhere(self.board == BUILDABLE)
        village_locations = [("build_village", tuple(x)) for x in buildable_locations if x[0] % 2 == 0]
        city_locations = [("build_city", tuple(x)) for x in buildable_locations if x[0] % 2 == 0]
        road_locations = [("build_road", tuple(x)) for x in buildable_locations if x[0] % 2 != 0]

        self.action_space = []
        self.action_space.extend(village_locations)     # First 54 cities
        self.action_space.extend(city_locations)        # Second 54 cities
        self.action_space.extend(road_locations)        # Road 20 locations

        self.action_space.extend([("trade", x) for x in permutations(self.players[0].resources, 2)])
        self.action_space.append(("pass", (-1, -1)))

    def create_board(self):
        if self.shuffle:
            r.shuffle(TILE_TYPE)
        self.tiles = [Tile(coords, tile_type) for (coords, tile_type) in zip(TILE_ID, TILE_TYPE)]
        if self.shuffle:
            r.shuffle(self.tiles)
        # Actual size 21, 23 (w, h)
        # 1 Represents buildable, even rows = villages/cities, odd rows = roads
        self.board = np.array(
            [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
             [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
             [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
             [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
             [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
             [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
             [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
             [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
             [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
             [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
             [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype='f')

        if 0 in TILE_NUMBER:  # 0 is inserted when bandit tile is generated
            TILE_NUMBER.remove(0)

        # Generate tile type and number
        for i, (tile, tile_number) in enumerate(zip(self.tiles, TILE_NUMBER)):
            '''for row_offset in range(-1, 2):
                for col_offset in range(-1, 2):
                    self.board[tile.row + row_offset][tile.col + col_offset] = tile.tile_type'''

            self.board[tile.row + 1][tile.col] = tile.tile_type
            self.board[tile.row - 1][tile.col] = tile.tile_type
            if tile.tile_type == 0:  # Bandit tile
                self.board[tile.row][tile.col] = 0
                TILE_NUMBER.insert(i, 0)  # Insert 0 so that tiles and tile number keep same length
            else:
                self.board[tile.row][tile.col] = tile_number
                tile.number = tile_number

    def print_board(self):
        for row in self.board:
            print("")
            for value in row:
                if not value % 1:
                    value = int(value)
                if value == 0 or value in []:
                    print(" " * 4, end='')
                else:
                    if len(str(value)) == 2:
                        print(value, end=' ' * 2)
                    elif len(str(value)) == 3:
                        print(value, end=' ' * 1)
                    else:
                        print(value, end=' ' * 3)
        print()

    # Dice throw and players gather resources
    def environment_step(self):
        dice_number = r.randint(1, 6) + r.randint(1, 6)

        if dice_number == 7:
            # TODO add robber
            # All players half their hands
            # Player move robber and steal resource
            pass
        else:
            for tile in self.tiles:     # Dictionary lookup?    Loop players and their villages instead?
                if tile.number == dice_number:
                    for row_offset, col_offset in [(1, -2), (-1, -2), (3, 0), (-3, 0), (1, 2), (-1, 2)]:
                        for player in self.players:
                            if self.board[tile.row + row_offset][tile.col + col_offset] == player.id:
                                if player.resources[tile.get_tile_type()] < 12:
                                    player.resources[tile.get_tile_type()] += 1

                            if self.board[tile.row + row_offset][tile.col + col_offset] == player.id + 0.5:
                                if player.resources[tile.get_tile_type()] < 11:
                                    player.resources[tile.get_tile_type()] += 2

    # Check if neighboring nodes are not built upon represented by 1
    def check_village_buildable(self, location):
        row, col = location

        values = []     # TODO try change so if value not == 1 return False for optimization
        padding_width = 2
        padded_board = np.pad(self.board, padding_width)
        if self.board[row][col] == BUILDABLE:
            if row % 4 == 0:    # 0 + 4n (difference in offset)
                values = [padded_board[row + row_offset + padding_width][col + col_offset + padding_width] for
                          (row_offset, col_offset) in [(-2, 0), (2, -2), (2, 2)]]

            elif row % 2 == 0:  # 2 + 4n
                values = [padded_board[row + row_offset + padding_width][col + col_offset + padding_width] for
                          (row_offset, col_offset) in [(2, 0), (-2, -2), (-2, 2)]]
        else:
            return False
        for value in values:
            if value == 0 or value == BUILDABLE:   # All neighboring village slots are not occupied
                continue
            else:   # Neighboring node occupied
                return False
        return True

    # Gets all buildable locations for roads and villages
    def get_buildable_locations(self, player):  # TODO Might belong in player
        buildable_road_locations = []
        buildable_village_locations = []
        padding_width = 2
        padded_board = np.pad(self.board, padding_width)

        for road in player.roads:
            row, col = road     # Maybe add redundant check board if place is = player.id
            if (row - 1) % 4 == 0:      # Diagonal road (1 + 4n)
                if road in LEFT_DIAGONAL:
                    road_offset = [(-2, -1), (0, -2), (0, 2), (2, 1)]
                    village_offset = [(1, 1), (-1, -1)]
                elif road in RIGHT_DIAGONAL:
                    road_offset = [(-2, 1), (0, -2), (0, 2), (2, -1)]
                    village_offset = [(-1, 1), (1, -1)]
                else:
                    raise ValueError(f'Road {road} location not correct')

                for row_offset, col_offset in road_offset:
                    if padded_board[row + row_offset + padding_width][col + col_offset + padding_width] == BUILDABLE:
                        buildable_road_locations.append((row + row_offset, col + col_offset))

                for row_offset, col_offset in village_offset:
                    if self.check_village_buildable((row + row_offset, col + col_offset)):
                        buildable_village_locations.append((row + row_offset, col + col_offset))

            elif (row - 1) % 2 == 0:    # Vertical road (3 + 4n)
                if self.check_village_buildable((row + 1, col)):
                    buildable_village_locations.append((row + 1, col))
                if self.check_village_buildable((row - 1, col)):
                    buildable_village_locations.append((row - 1, col))
                for row_offset, col_offset in [(-2, -1), (-2, 1), (2, -1), (2, 1)]:  # adjacent roads
                    if padded_board[row + row_offset + padding_width][col + col_offset + padding_width] == BUILDABLE:
                        buildable_road_locations.append((row + row_offset, col + col_offset))

        for village in player.villages:   # TODO temporary solution sometimes village gets built twice
            if village in buildable_village_locations:
                buildable_village_locations.remove(village)
            if village in player.cities:
                player.villages.remove(village)
        for road in player.roads:
            if road in buildable_road_locations:
                buildable_road_locations.remove(road)

        return list(set(buildable_road_locations)), list(set(buildable_village_locations))

    def step(self, action, location, player_id):
        row, col = location
        # TODO change so that players gets updated here as well (not in Agent file)
        if action == "build_village" or action == "build_road" and self.board[row][col] == BUILDABLE:
            self.board[row][col] = player_id    # TODO add longest road check
        elif action == "build_city" and self.board[row][col] == player_id:
            self.board[row][col] = player_id + 0.5
        return self

    def last(self):
        for player in self.players:
            if player.score >= 10:
                return True
        return False

    def free_build_village(self):
        buildable_locations = [tuple(x) for x in np.argwhere(self.board == 1)
                               if x[0] % 2 == 0 and self.check_village_buildable(x)]

        return buildable_locations

    def get_adjacent_roads(self, village_location):
        row, col = village_location
        padding_width = 1
        buildable_road_locations = []
        padded_board = np.pad(self.board, padding_width)
        if row % 4 == 0:
            for row_offset, col_offset in [(-1, 0), (1, -1), (1, 1)]:
                if padded_board[row + row_offset + padding_width][col + col_offset + padding_width] == BUILDABLE:
                    buildable_road_locations.append((row + row_offset, col + col_offset))

        elif row % 2 == 0:
            for row_offset, col_offset in [(-1, 1), (-1, -1), (1, 0)]:
                if padded_board[row + row_offset + padding_width][col + col_offset + padding_width] == BUILDABLE:
                    buildable_road_locations.append((row + row_offset, col + col_offset))

        return buildable_road_locations

    def reset_env(self):
        self.create_board()
        r.shuffle(self.players)

    def reward(self, agent):
        if self.last():
            result = sorted(self.players, key=lambda x: x.score, reverse=True)
            placement = result.index(agent)
            if result[-1] == agent:
                return -1.0

            if placement == 0:
                return 1
            elif placement == 1:
                return 0.7
            elif placement == 2:
                return 0.3
            elif placement == 3: # TODO Trying -0.1
                return -1.0
            else:
                return 0
        else:
            # Add small intermediate reward for building village/city, to counter excessive roads
            return 0



'''width, height = 200, 200
gridDisplay = pygame.display.set_mode((width, height))
pygame.display.get_surface().fill((width, height, 200))

grid_node_width = 10
grid_node_height = 10


def create_square(x, y, color):
    pygame.draw.rect(gridDisplay, color, [x, y, grid_node_width, grid_node_height])


def visualize_grid():
    y = 0
    for row in env.board:
        x = 0
        for item in row:
            if item == 0:
                create_square(x, y, (255, 255, 255))
            else:
                create_square(x, y, (item*10, 0, 0))
            x += grid_node_width
        y += grid_node_height

    pygame.display.update()


visualize_grid()

while True:
    pass

'''