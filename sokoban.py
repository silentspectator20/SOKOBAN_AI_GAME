import heapq
from collections import deque
import os
import numpy as np
import math

# State space
class StateSpace:
    n = 0
    
    def __init__(self, action, gval, parent):
        self.action = action
        self.gval = gval
        self.parent = parent
        self.index = StateSpace.n
        StateSpace.n = StateSpace.n + 1

    def successors(self):
        raise Exception("Must be overridden in subclass.")

    def hashable_state(self):
        raise Exception("Must be overridden in subclass.")

    def print_state(self):
        raise Exception("Must be overridden in subclass.")

    def print_path(self):
        s = self
        states = []
        while s:
            states.append(s)
            s = s.parent
        states.pop().print_state()
        while states:
            print(" ==> ", end="")
            states.pop().print_state()
        print("")
 
    def has_path_cycle(self):
        s = self.parent
        hc = self.hashable_state()
        while s:
            if s.hashable_state() == hc:
                return True
            s = s.parent
        return False

_DEPTH_FIRST = 0
_BREADTH_FIRST = 1
_BEST_FIRST = 2
_ASTAR = 3
_UCS = 4

_SUM_HG = 0
_H = 1
_G = 2
_C = 3

_CC_NONE = 0 # no cycle checking
_CC_PATH = 1 # path checking only
_CC_FULL = 2 # full cycle checking

def _zero_hfn(state):
    return 0

def _fval_function(state):
    return state.hval

class sNode:
    n = 0
    lt_type = _SUM_HG
    
    def __init__(self, state, hval, fval_function):
        self.state = state
        self.hval = hval
        self.gval = state.gval
        self.index = sNode.n
        self.fval_function = fval_function
        sNode.n = sNode.n + 1

    def __lt__(self, other):     
        if sNode.lt_type == _SUM_HG:
            if (self.gval+self.hval) == (other.gval+other.hval):
                return self.gval > other.gval
            else: return ((self.gval+self.hval) < (other.gval+other.hval))
        if sNode.lt_type == _G:
            return self.gval < other.gval
        if sNode.lt_type == _H:
            return self.hval < other.hval    
        if sNode.lt_type == _C:  
            return self.fval_function(self) <  other.fval_function(other)          
        print('sNode class has invalid comparator setting!')
        return self.gval < other.gval

class Open:
    def __init__(self, search_strategy):
        if search_strategy == _DEPTH_FIRST:
            self.open = []
            self.insert = self.open.append
            self.extract = self.open.pop
        elif search_strategy == _BREADTH_FIRST:
            self.open = deque()
            self.insert = self.open.append
            self.extract = self.open.popleft
        elif search_strategy == _UCS:
            self.open = []
            sNode.lt_type = _G
            self.insert = lambda node: heapq.heappush(self.open, node)
            self.extract = lambda: heapq.heappop(self.open)            
        elif search_strategy == _BEST_FIRST:
            self.open = []
            sNode.lt_type = _H
            self.insert = lambda node: heapq.heappush(self.open, node)
            self.extract = lambda: heapq.heappop(self.open)
        elif search_strategy == _ASTAR:
            self.open = []
            sNode.lt_type = _SUM_HG
            self.insert = lambda node: heapq.heappush(self.open, node)
            self.extract = lambda: heapq.heappop(self.open)          

    def empty(self): 
        return not self.open

    def print_open(self):
        print("{", end="")
        if len(self.open) == 1: 
            print("   <S{}:{}:{}, g={}, h={}, f=g+h={}>".format(self.open[0].state.index, self.open[0].state.action, self.open[0].state.hashable_state(), self.open[0].gval, self.open[0].hval, self.open[0].gval+self.open[0].hval), end="")
        else:
            for nd in self.open:
                print("   <S{}:{}:{}, g={}, h={}, f=g+h={}>".format(nd.state.index, nd.state.action, nd.state.hashable_state(), nd.gval, nd.hval, nd.gval+nd.hval), end="")
        print("}")

class SearchEngine:
    def __init__(self, strategy = 'depth_first', cc_level = 'default'):
        self.set_strategy(strategy, cc_level)
        self.trace = 0

    def initStats(self):
        sNode.n = 0
        StateSpace.n = 1    
        self.cycle_check_pruned = 0
        self.cost_bound_pruned = 0

    def trace_on(self, level = 1):
        self.trace = level

    def trace_off(self):
        self.trace = 0

    def set_strategy(self, s, cc = 'default'):
        if not s in ['depth_first', 'breadth_first', 'ucs', 'best_first', 'astar']:
            print('Unknown search strategy specified:', s)
            print("Must be one of 'depth_first', 'ucs', 'breadth_first', 'best_first' or 'astar'")
        elif not cc in ['default', 'none', 'path', 'full']:
            print('Unknown cycle check level', cc)
            print( "Must be one of ['default', 'none', 'path', 'full']")
        else:
            if cc == 'default' :
                if s == 'depth_first' :
                    self.cycle_check = _CC_PATH
                else:
                    self.cycle_check = _CC_FULL
            elif cc == 'none': self.cycle_check = _CC_NONE
            elif cc == 'path': self.cycle_check = _CC_PATH
            elif cc == 'full': self.cycle_check = _CC_FULL

            if   s == 'depth_first'  : self.strategy = _DEPTH_FIRST
            elif s == 'breadth_first': self.strategy = _BREADTH_FIRST
            elif s == 'ucs' : self.strategy = _UCS               
            elif s == 'best_first'   : self.strategy = _BEST_FIRST
            elif s == 'astar'        : self.strategy = _ASTAR            

    def get_strategy(self):
        if   self.strategy == _DEPTH_FIRST    : rval = 'depth_first'
        elif self.strategy == _BREADTH_FIRST  : rval = 'breadth_first'
        elif self.strategy == _BEST_FIRST     : rval = 'best_first' 
        elif self.strategy == _UCS          : rval = 'ucs' 
        elif self.strategy == _ASTAR          : rval = 'astar'     

        rval = rval + ' with '
        if   self.cycle_check == _CC_NONE : rval = rval + 'no cycle checking'
        elif self.cycle_check == _CC_PATH : rval = rval + 'path checking'
        elif self.cycle_check == _CC_FULL : rval = rval + 'full cycle checking'
        return rval

    def init_search(self, initState, goal_fn, heur_fn=_zero_hfn, fval_function=_fval_function):
        self.initStats()
        if self.trace:
            print("   TRACE: Search Strategy: ", self.get_strategy())
            print("   TRACE: Initial State:", end="")
            initState.print_state()
        self.open = Open(self.strategy)
        node = sNode(initState, heur_fn(initState), fval_function)      
        if self.cycle_check == _CC_FULL:
            self.cc_dictionary = dict() 
            self.cc_dictionary[initState.hashable_state()] = initState.gval
        self.open.insert(node)
        self.fval_function = fval_function
        self.goal_fn = goal_fn
        self.heur_fn = heur_fn

    def search(self, timebound=10, costbound=None):
        goal_node = []
        self.search_start_time = os.times()[0]
        self.search_stop_time = None
        if timebound:
            self.search_stop_time = self.search_start_time + timebound
        goal_node = self._searchOpen(self.goal_fn, self.heur_fn, self.fval_function, costbound)
        if goal_node:
            total_search_time = os.times()[0] - self.search_start_time
            print("Solution Found with cost of {} in search time of {} sec".format(goal_node.gval, total_search_time))
            print("Nodes expanded = {}, states generated = {}, states cycle check pruned = {}, states cost bound pruned = {}".format(sNode.n, StateSpace.n, self.cycle_check_pruned, self.cost_bound_pruned))
            return goal_node.state
        else:
            total_search_time = os.times()[0] - self.search_start_time            
            print("Search Failed! No solution found.")
            print("Nodes expanded = {}, states generated = {}, states cycle check pruned = {}, states cost bound pruned = {}".format(sNode.n, StateSpace.n, self.cycle_check_pruned, self.cost_bound_pruned))
            return False

    def _searchOpen(self, goal_fn, heur_fn, fval_function, costbound):
        if self.trace:
            print("   TRACE: Initial OPEN: ", self.open.print_open())
            if self.cycle_check == _CC_FULL:
                print("   TRACE: Initial CC_Dict:", self.cc_dictionary)
        while not self.open.empty():
            node = self.open.extract()
            if self.trace:
                print("   TRACE: Next State to expand: <S{}:{}:{}, g={}, h={}, f=g+h={}>".format(node.state.index, node.state.action, node.state.hashable_state(), node.gval, node.hval, node.gval + node.hval))
                if node.state.gval != node.gval:
                    print("ERROR: Node gval not equal to state gval!")
                        
            if goal_fn(node.state):
              return node

            if self.search_stop_time: 
              if os.times()[0] > self.search_stop_time:     
                print("TRACE: Search has exceeeded the time bound provided.")
                return False
            if self.trace:
                if self.cycle_check == _CC_FULL: 
                  print("   TRACE: CC_dict gval={}, node.gval={}".format(self.cc_dictionary[node.state.hashable_state()], node.gval))

            if self.cycle_check == _CC_FULL and self.cc_dictionary[node.state.hashable_state()] < node.gval:
                continue

            successors = node.state.successors()
            if self.trace:
                print("   TRACE: Expanding Node. Successors = {", end="")
                for ss in successors:                  
                    print("<S{}:{}:{}, g={}, h={}, f=g+h={}>, ".format(ss.index, ss.action, ss.hashable_state(), ss.gval, heur_fn(ss), ss.gval+heur_fn(ss)), end="")                    
                print("}")

            for succ in successors:
                hash_state = succ.hashable_state()
                if self.trace > 1: 
                  if self.cycle_check == _CC_FULL and hash_state in self.cc_dictionary:
                      print("   TRACE: Already in CC_dict, CC_dict gval={}, successor state gval={}".format(
                        self.cc_dictionary[hash_state], succ.gval))   
                if self.trace > 1:
                    print("   TRACE: Successor State:", end="")
                    succ.print_state()
                    print("   TRACE: Heuristic Value:", heur_fn(succ))

                    if self.cycle_check == _CC_FULL and hash_state in self.cc_dictionary:
                        print("   TRACE: Already in CC_dict, CC_dict gval={}, successor state gval={}".format(self.cc_dictionary[hash_state], succ.gval))

                    if self.cycle_check == _CC_PATH and succ.has_path_cycle():
                        print("   TRACE: On cyclic path")

                prune_succ = (self.cycle_check == _CC_FULL and hash_state in self.cc_dictionary and succ.gval > self.cc_dictionary[hash_state]) or ( self.cycle_check == _CC_PATH and succ.has_path_cycle() )

                if prune_succ :
                    self.cycle_check_pruned = self.cycle_check_pruned + 1
                    if self.trace > 1:
                        print(" TRACE: Successor State pruned by cycle checking")
                        print("\n")          
                    continue

                succ_hval = heur_fn(succ)
                if costbound is not None and (succ.gval > costbound[0] or succ_hval > costbound[1] or succ.gval + succ_hval > costbound[2]) : 
                    self.cost_bound_pruned = self.cost_bound_pruned + 1
                    if self.trace > 1:
                      print(" TRACE: Successor State pruned, over current cost bound of {}", costbound)
                      print("\n") 
                    continue                    

                self.open.insert(sNode(succ, succ_hval, node.fval_function))

                if self.trace > 1:
                    print(" TRACE: Successor State added to OPEN")
                    print("\n")
                if self.cycle_check == _CC_FULL:
                    self.cc_dictionary[hash_state] = succ.gval

        return False

# specializion of the StateSpace Class that is tailored to the game of Sokoban
class SokobanState(StateSpace):

    def __init__(self, action, gval, parent, width, height, robot, boxes, storage, obstacles, restrictions=None, box_colours=None, storage_colours=None):
        StateSpace.__init__(self, action, gval, parent)
        self.width = width
        self.height = height
        self.robot = robot
        self.boxes = boxes
        self.storage = storage
        self.obstacles = obstacles
        self.restrictions = restrictions
        self.box_colours = box_colours
        self.storage_colours = storage_colours

    def successors(self):
        successors = []
        transition_cost = 1

        for direction in (UP, RIGHT, DOWN, LEFT):
            new_location = direction.move(self.robot)
            
            if new_location[0] < 0 or new_location[0] >= self.width:
                continue
            if new_location[1] < 0 or new_location[1] >= self.height:
                continue
            if new_location in self.obstacles:
                continue
            
            new_boxes = dict(self.boxes)

            if new_location in self.boxes:
                new_box_location = direction.move(new_location)
                
                if new_box_location[0] < 0 or new_box_location[0] >= self.width:
                    continue
                if new_box_location[1] < 0 or new_box_location[1] >= self.height:
                    continue
                if new_box_location in self.obstacles:
                    continue
                if new_box_location in new_boxes:
                    continue
                
                index = new_boxes.pop(new_location)
                new_boxes[new_box_location] = index
            
            new_robot = tuple(new_location)

            new_state = SokobanState(action=direction.name, gval=self.gval + transition_cost, parent=self, width=self.width, height=self.height, robot=new_robot, boxes=new_boxes, storage=self.storage, obstacles=self.obstacles, restrictions=self.restrictions, box_colours=self.box_colours, storage_colours=self.storage_colours)
            successors.append(new_state)

        return successors

    def hashable_state(self):
        return hash((self.robot, frozenset(self.boxes.items())))

    def state_string(self):
        disable_terminal_colouring = False
        fg_colours = {
            'red': '\033[31m',
            'cyan': '\033[36m',
            'blue': '\033[34m',
            'green': '\033[32m',
            'magenta': '\033[35m',
            'yellow': '\033[33m',
            'normal': '\033[0m'
        }
        bg_colours = {
            'red': '\033[41m',
            'cyan': '\033[46m',
            'blue': '\033[44m',
            'green': '\033[42m',
            'magenta': '\033[45m',
            'yellow': '\033[43m',
            'normal': '\033[0m'
        }
        map = []
        for y in range(0, self.height):
            row = []
            for x in range(0, self.width):
                row += [' ']
            map += [row]
        if self.storage_colours:
            if disable_terminal_colouring:
                for storage_point in self.storage:
                    map[storage_point[1]][storage_point[0]] = self.storage_colours[self.storage[storage_point]][0:1].upper()
            else:
                for storage_point in self.storage:
                    map[storage_point[1]][storage_point[0]] = bg_colours[self.storage_colours[self.storage[storage_point]]] + '.' + bg_colours['normal']
        else:
            for (i, storage_point) in enumerate(self.storage):
                map[storage_point[1]][storage_point[0]] = '.'
        for obstacle in self.obstacles:
            map[obstacle[1]][obstacle[0]] = '#'
        map[self.robot[1]][self.robot[0]] = '?'
        if self.box_colours:
            if disable_terminal_colouring:
                for box in self.boxes:
                    if box in self.storage:
                        if self.restrictions is None or box in self.restrictions[self.boxes[box]]:
                            map[box[1]][box[0]] = '$'
                        else:
                            map[box[1]][box[0]] = 'x'
                    else:
                        map[box[1]][box[0]] = self.box_colours[self.boxes[box]][0:1].lower()
            else:
                for box in self.boxes:
                    if box in self.storage:
                        if self.restrictions is None or box in self.restrictions[self.boxes[box]]:
                            map[box[1]][box[0]] = bg_colours[self.storage_colours[self.storage[box]]] + fg_colours[self.box_colours[self.boxes[box]]] + '$' + bg_colours['normal']
                        else:
                            map[box[1]][box[0]] = bg_colours[self.storage_colours[self.storage[box]]] + fg_colours[self.box_colours[self.boxes[box]]] + 'x' + bg_colours['normal']
                    else:
                        map[box[1]][box[0]] = fg_colours[self.box_colours[self.boxes[box]]] + '*' + fg_colours['normal']
        else:
            for box in self.boxes:
                if box in self.storage:
                    if self.restrictions is None or box in self.restrictions[self.boxes[box]]:
                        map[box[1]][box[0]] = '$'
                    else:
                        map[box[1]][box[0]] = 'x'
                else:
                    map[box[1]][box[0]] = '*'

        for y in range(0, self.height):
            map[y] = ['#'] + map[y]
            map[y] = map[y] + ['#']
        map = ['#' * (self.width + 2)] + map
        map = map + ['#' * (self.width + 2)]

        s = ''
        for row in map:
            for char in row:
                s += char
            s += '\n'

        return s        

    def print_state(self):      
        print("ACTION was " + self.action)      
        print(self.state_string())

def sokoban_goal_state(state):
  if state.restrictions is None:
    for box in state.boxes:
      if box not in state.storage:
        return False
    return True
  for box in state.boxes:
    if box not in state.restrictions[state.boxes[box]]:
      return False
  return True

def generate_coordinate_rect(x_start, x_finish, y_start, y_finish):
    coords = []
    for i in range(x_start, x_finish):
        for j in range(y_start, y_finish):
            coords.append((i, j))
    return coords

PROBLEMS = (
    SokobanState("START", 0, None, 4, 4, # dimensions
                 (0, 3), #robot
                 {(1, 2): 0, (1, 1): 1}, #boxes 
                 {(2, 1): 0, (2, 2): 1}, #storage
                 frozenset(((0, 0), (1, 0), (3, 3))), #obstacles
                 (frozenset(((2, 1),)), frozenset(((2, 2),))), #restrictions,
                 {0: 'cyan', 1: 'magenta'}, #box colours
                 {0: 'cyan', 1: 'magenta'} #storage colours
                 ),
    SokobanState("START", 0, None, 6, 4, # dimensions
             (5, 3), #robot
             {(1, 1): 0, (3, 1): 1}, #boxes 
             {(2, 0): 0, (2, 2): 1}, #storage
             frozenset(((2, 1), (0, 0), (5, 0), (0, 3), (1, 3), (2, 3), (3, 3))), #obstacles
             (frozenset(((2, 0),)), frozenset(((2, 2),))), #restrictions,
             {0: 'cyan', 1: 'magenta'}, #box colours
             {0: 'cyan', 1: 'magenta'} #storage colours
             ),
    SokobanState("START", 0, None, 5, 4, # dimensions
             (0, 3), #robot
             {(2, 1): 0, (3, 1): 1}, #boxes 
             {(2, 1): 0, (3, 1): 1}, #storage
             frozenset(((0, 0), (4, 0), (2, 3), (3, 3), (4, 3))), #obstacles
             (frozenset(((3, 1),)), frozenset(((2, 1),))), #restrictions,
             {0: 'cyan', 1: 'magenta'}, #box colours
             {1: 'cyan', 0: 'magenta'} #storage colours
             ),
    SokobanState("START", 0, None, 5, 5, # dimensions
                 (2, 1), # robot
                 {(1, 1): 0, (1, 3): 1, (3, 1): 2, (3, 3): 3}, #boxes 
                 {(0, 0): 0, (0, 4): 1, (4, 0): 2, (4, 4): 3}, #storage
                 frozenset(((1, 0), (2, 0), (3, 0), (1, 4), (2, 4), (3, 4))), #obstacles
                 None #restrictions
                 ),
    SokobanState("START", 0, None, 5, 5, # dimensions
                 (4, 0), #robot
                 {(3, 1): 0, (3, 2): 1, (3, 3): 2}, #boxes 
                 {(0, 0): 0, (0, 2): 1, (0, 4): 2}, #storage
                 frozenset(((2, 0), (2, 1), (2, 3), (2, 4))), #obstacles
                 None #restrictions
                 ),
    SokobanState("START", 0, None, 5, 5, # dimensions
                 (4, 0), #robot
                 {(3, 1): 0, (3, 2): 1, (3, 3): 2}, #boxes 
                 {(0, 0): 0, (0, 2): 1, (0, 4): 2}, #storage
                 frozenset(((2, 0), (2, 1), (2, 3), (2, 4))), #obstacles
                 None #restrictions
                 ),
    SokobanState("START", 0, None, 6, 4, # dimensions
         (5, 3), #robot
         {(3, 1): 0, (2, 2): 1, (3, 2): 2, (4, 2): 3}, #boxes 
         {(0, 0): 0, (2, 0): 1, (1, 0): 2, (1, 1): 3}, #storage
         frozenset((generate_coordinate_rect(4, 6, 0, 1)
                   + generate_coordinate_rect(0, 3, 3, 4))), #obstacles
         (frozenset(((0, 0),)), frozenset(((2, 0),)), frozenset(((1, 0),)), frozenset(((1, 1),))), #restrictions,
         {0: 'cyan', 1: 'magenta', 2: 'yellow', 3: 'red'}, #box colours
         {0: 'cyan', 1: 'magenta', 2: 'yellow', 3: 'red'} #storage colours
         ),
    SokobanState("START", 0, None, 6, 4, # dimensions
         (5, 3), #robot
         {(3, 1): 0, (2, 2): 1, (3, 2): 2, (4, 2): 3}, #boxes 
         {(0, 0): 0, (2, 0): 1, (1, 0): 2, (1, 1): 3}, #storage
         frozenset((generate_coordinate_rect(4, 6, 0, 1)
                   + generate_coordinate_rect(0, 3, 3, 4))), #obstacles
         (frozenset(((0, 0),)), frozenset(((2, 0),)), frozenset(((1, 0),)), frozenset(((0, 0), (2, 0), (1, 0), (1, 1),))), #restrictions,
         {0: 'cyan', 1: 'magenta', 2: 'yellow', 3: 'normal'}, #box colours
         {0: 'cyan', 1: 'magenta', 2: 'yellow', 3: 'red'} #storage colours
         ),
    SokobanState("START", 0, None, 8, 6, # dimensions
         (1, 2), #robot
         {(1, 3): 0, (2, 3): 1, (3, 3): 2, (4, 3): 3, (5, 3): 4}, #boxes 
         {(7, 0): 0, (7, 1): 1, (7, 2): 2, (7, 3): 3, (7, 4): 4}, #storage
         frozenset((generate_coordinate_rect(0, 7, 0, 2) + [(0, 2), (6, 2), (7, 5)]
         + generate_coordinate_rect(0, 5, 5, 6))), #obstacles
         (frozenset(((7, 0),)), frozenset(((7, 1),)), frozenset(((7, 2),)), frozenset(((7, 3),)), frozenset(((7, 4),))), #restrictions,
         {0: 'cyan', 1: 'magenta', 2: 'yellow', 3: 'red', 4: 'green'}, #box colours
         {0: 'cyan', 1: 'magenta', 2: 'yellow', 3: 'red', 4: 'green'} #storage colours
         ),
    SokobanState("START", 0, None, 6, 5, # dimensions
         (5, 2), #robot
         {(3, 1): 0, (3, 2): 1, (3, 3): 2, (4, 2): 3}, #boxes 
         {(1, 2): 0, (2, 2): 1, (3, 2): 2, (0, 2): 3}, #storage
         frozenset((generate_coordinate_rect(4, 6, 0, 1)
                    + generate_coordinate_rect(3, 6, 4, 5))
                    + [(1, 1), (1, 3)]), #obstacles
         (frozenset(((1, 2),)), frozenset(((2, 2),)), frozenset(((3, 2),)), frozenset(((0, 2),)), frozenset(((7, 4),))), #restrictions,
         {0: 'cyan', 1: 'magenta', 2: 'yellow', 3: 'red', 4: 'green'}, #box colours
         {0: 'cyan', 1: 'magenta', 2: 'yellow', 3: 'red', 4: 'green'} #storage colours
         ),
    )
 
class Direction():
    def __init__(self, name, delta):
        self.name = name
        self.delta = delta
    
    def __hash__(self):
        return hash(self.name)
    
    def __str__(self):
        return str(self.name)
    
    def __repr__(self):
        return self.__str__()
    
    def move(self, location):
        return (location[0] + self.delta[0], location[1] + self.delta[1])

# Global Directions
UP = Direction("up", (0, -1))
RIGHT = Direction("right", (1, 0))
DOWN = Direction("down", (0, 1))
LEFT = Direction("left", (-1, 0))

def heur_displaced(state):
  count = 0
  for box in state.boxes:
    if box not in state.storage:
      count += 1
  return count

def absolute(num):
  if num < 0:
    return num * -1
  else:
    return num

def heur_manhattan_distance(state):
    h=0 
    for box in state.boxes:
      box_num=state.boxes[box] 
      if state.restrictions == None:
        min_distance = float("inf") 
        for storage in state.storage:
          distance = absolute(storage[0]-box[0])+absolute(storage[1]-box[1])
          if distance <= min_distance:
            min_distance=distance
      elif (len(state.restrictions[box_num])!=1): 
          min_distance = float("inf") 
          for i in range(len(state.restrictions[box_num])): 
            restriction = list(state.restrictions[box_num])
            restriction = restriction[i]
            distance = absolute(restriction[0]-box[0])+absolute(restriction[1]-box[1])
            if distance <= min_distance:
              min_distance=distance
              
      else: 
          restriction = list(state.restrictions[box_num])
          restriction = restriction[0]
          distance = absolute(restriction[0]-box[0])+absolute(restriction[1]-box[1])
          min_distance = distance
      h+=min_distance  
      min_distance = float("inf") 
    return h

def heur_euclidean_distance(state): 
    h=0
    for box in state.boxes:
      box_num=state.boxes[box] 
      if state.restrictions == None:
        min_distance = float("inf") 
        for storage in state.storage:
          distance = math.sqrt(abs(storage[0]-box[0])**2+abs(storage[1]-box[1])**2)
          if distance <= min_distance:
            min_distance=distance
      elif (len(state.restrictions[box_num])!=1): 
          min_distance = float("inf") 
          for i in range(len(state.restrictions[box_num])): 
            restriction = list(state.restrictions[box_num])
            restriction = restriction[i]
            distance = math.sqrt(abs(restriction[0]-box[0])**2+abs(restriction[1]-box[1])**2)
            if distance <= min_distance:
              min_distance=distance      
      else: 
          restriction = list(state.restrictions[box_num])
          restriction = restriction[0]
          distance = math.sqrt(abs(restriction[0]-box[0])**2+abs(restriction[1]-box[1])**2)
          min_distance = distance
      h+=min_distance 
      min_distance = float("inf") 
    return h

def heur_robot_to_goal(state):
    h=0
    distance = float('inf')
    min_distance = float('inf')
    
    x=state.robot[0]
    y=state.robot[1]
    
    for storage in state.storage:
      if state.robot not in storage:
        distance = abs(storage[0]-x) + abs(storage[1]-y)
        if distance < min_distance:
          min_distance = distance
    return min_distance

# sums the distances from any box to any storage
def L2Norm(state): 
    min_distance = float('inf')
    distance = 0
    for box in state.boxes:
      for storage in state.storage:
        distance += ((storage[0]-box[0])**2 + (storage[1]-box[1])**2)**0.5
    return distance

# min distance
def L2Norm1(state):
    min_distance = float('inf')
    for box in state.boxes:
      for storage in state.storage:
        distance = ((storage[0]-box[0])**2 + (storage[1]-box[1])**2)**0.5
        if distance < min_distance:
          min_distance = distance
    return min_distance

problem_number = 0

def heur_alternate(state):
    global problem_number
    global last_heuristic
    global prev_state
    global look_up
    key = ''
    if state.parent == None:
      look_up = {}
      problem_number += 1
    try:
      if state.boxes == prev_state:
        return last_heuristic
    except:
      pass
    
    prev_state = state.boxes
    height = state.height
    width = state.width
    
    for box in state.boxes:
      key = key + str(box)
    if key in look_up:
      last_heuristic = look_up[key]
      return last_heuristic
    
    manhattan_distance = 0
    for box in state.boxes:
      if state.restrictions == None: 
        result = deadblock_check(box,state,height,width,state.storage,key)
        if result == float('inf'):
          return result
      else:
        index = state.boxes[box] 
        result = deadblock_check(box,state,height,width,state.restrictions[index],key)
        if result == float('inf'):
          return result
    
    last_heuristic = result
    look_up[key] = last_heuristic 
    return result

def deadblock_check(box,state,height,width,goal_restriction,key):
  global last_heuristic
  global prev_state
  global look_up

  if box not in goal_restriction: 
    x=box[0]
    y=box[1]
    
    left_obs = False
    right_obs = False
    top_obs = False
    bottom_obs = False
    
    left = (x-1,y)
    right = (x+1,y)
    top = (x,y-1)
    bottom = (x,y+1)
    
    if top in state.obstacles or top[1] < 0 or top in state.boxes:
      top_obs = True
    if bottom in state.obstacles or bottom[1] > (height - 1) or bottom in state.boxes:
      bottom_obs = True
    if left in state.obstacles or left[0] < 0 or left in state.boxes:
      left_obs = True
    if right in state.obstacles or right[0] > (width -1) or right in state.boxes:
      right_obs = True
    
    
    safe = False 
    
    if box[0] == 0 or box[0] == (width-1): 
      if top_obs or bottom_obs: 
        last_heuristic = float('inf')
        look_up[key] = last_heuristic
        return float('inf')
        
      if state.restrictions != None:
        goal_index = state.boxes[box]
        corr_restriction = state.restrictions[goal_index]
        for restriction in corr_restriction: 
          if box[0] == 0 and restriction[0] == 0:
            safe = True
          if box[0] == (width-1) and restriction[0] == (width-1):
            safe = True
        if safe == False:
          last_heuristic = float('inf')
          look_up[key] = last_heuristic
          return float('inf')
          
      elif state.restrictions == None: 
        for storage in state.storage:
          if box[0] == 0 and storage[0] == 0:
            safe = True
          if box[0] == (width-1) and storage[0] == (width-1):
            safe = True
        if safe == False:
          last_heuristic = float('inf')
          look_up[key] = last_heuristic
          return float('inf')

    elif box[1]==0 or box[1] == (height-1): 
      if left_obs or right_obs:
        last_heuristic = float('inf')
        look_up[key] = last_heuristic
        return float('inf')
        
      if state.restrictions != None:  
        goal_index = state.boxes[box]
        corr_restriction = state.restrictions[goal_index]
        for restriction in corr_restriction:
          if box[1] == 0 and restriction[1] == 0:
            safe = True
          if box[1] == (height-1) and restriction[1] == (height-1):
            safe = True
        if safe == False:
          last_heuristic = float('inf')
          look_up[key] = last_heuristic
          return float('inf')
          
      elif state.restrictions == None:
        for storage in state.storage:
          if box[1] == 0 and storage[1] == 0:
            safe = True
          if box[1] == (height-1) and storage[1] == (height-1):
            safe = True
        if safe == False:
          last_heuristic = float('inf')
          look_up[key] = last_heuristic
          return float('inf')
 
    left_obs = False
    right_obs = False
    top_obs = False
    bottom_obs = False
    
    if top in state.obstacles or top[1] < 0:
      top_obs = True
    if bottom in state.obstacles or bottom[1] > (height - 1):
      bottom_obs = True
    if left in state.obstacles or left[0] < 0:
      left_obs = True
    if right in state.obstacles or right[0] > (width -1):
      right_obs = True

    if top_obs and right_obs:
      last_heuristic = float('inf')
      look_up[key] = last_heuristic
      return float('inf')
    elif right_obs and bottom_obs:
      last_heuristic = float('inf')
      look_up[key] = last_heuristic
      return float('inf')
    elif bottom_obs and left_obs:
      last_heuristic = float('inf')
      look_up[key] = last_heuristic
      return float('inf')
    elif left_obs and top_obs:
      last_heuristic = float('inf')
      look_up[key] = last_heuristic
      return float('inf') 
    
    left_obs = False
    right_obs = False
    top_obs = False
    bottom_obs = False
    
    if top in state.obstacles or top[1] < 0 or top in state.boxes:
      top_obs = True
    if bottom in state.obstacles or bottom[1] > (height - 1) or bottom in state.boxes:
      bottom_obs = True
    if left in state.obstacles or left[0] < 0 or left in state.boxes:
      left_obs = True
    if right in state.obstacles or right[0] > (width -1) or right in state.boxes:
      right_obs = True
      
    bottom_left_corner = (x-1,y+1)
    bottom_right_corner = (x+1,y+1)
    top_left_corner = (x-1,y-1)
    top_right_corner = (x+1,y-1)
   
    if left_obs and bottom_obs and bottom_left_corner in state.obstacles:
      last_heuristic = float('inf')
      look_up[key] = last_heuristic
      return float('inf')
    if right_obs and bottom_obs and bottom_right_corner in state.obstacles:
      last_heuristic = float('inf')
      look_up[key] = last_heuristic
      return float('inf')
    if left_obs and top_obs and top_left_corner in state.obstacles:
      last_heuristic = float('inf')
      look_up[key] = last_heuristic
      return float('inf')
    if right_obs and top_obs and top_right_corner in state.obstacles:
      last_heuristic = float('inf')
      look_up[key] = last_heuristic
      return float('inf')
    
  return 0

def fval_function(sN, weight):
    return sN.gval + weight * sN.hval

# "#" is a wall,
# " " is a free space,
# "$" is a box,
# "." is a goal place,
# "*" is a boxes placed on a goal,
# "?" is for Sokoban and
# "$" is for Sokoban on a goal.

if __name__ == "__main__":
  solved = 0; unsolved = []; counter = 0; percent = 0; timebound = 2; #2 second time limit for each problem
  print("*************************************")  
  print("Running A-star")     

  for i in range(0, 10): 

    print("*************************************")  
    print("PROBLEM {}".format(i))
    
    s0 = PROBLEMS[i] 

    se = SearchEngine('astar', 'full')
    se.init_search(s0, goal_fn=sokoban_goal_state, heur_fn=heur_displaced)
    final = se.search(timebound)

    if final:
      final.print_path()
      solved += 1
    else:
      unsolved.append(i)    
    counter += 1

  if counter > 0:  
    percent = (solved/counter)*100

  print("*************************************")  
  print("{} of {} problems ({} %) solved in less than {} seconds.".format(solved, counter, percent, timebound))  
  print("Problems that remain unsolved in the set are Problems: {}".format(unsolved))      
  print("*************************************") 
  print("*************************************") 
  print("*************************************") 
  print("*************************************") 

  solved = 0; unsolved = []; counter = 0; percent = 0; timebound = 2; #2 second time limit for each problem
  print("*************************************")  
  print("Running Depth-first")     

  for i in range(0, 10): 

    print("*************************************")  
    print("PROBLEM {}".format(i))
    
    s0 = PROBLEMS[i] 

    se = SearchEngine('depth_first', 'full')
    se.init_search(s0, goal_fn=sokoban_goal_state, heur_fn=heur_displaced)
    final = se.search(timebound)

    if final:
      final.print_path()
      solved += 1
    else:
      unsolved.append(i)    
    counter += 1

  if counter > 0:  
    percent = (solved/counter)*100

  print("*************************************")  
  print("{} of {} problems ({} %) solved in less than {} seconds.".format(solved, counter, percent, timebound))  
  print("Problems that remain unsolved in the set are Problems: {}".format(unsolved))      
  print("*************************************") 
  print("*************************************") 
  print("*************************************") 
  print("*************************************") 

  solved = 0; unsolved = []; counter = 0; percent = 0; timebound = 2; #2 second time limit for each problem
  print("*************************************")  
  print("Running Breadth-first")     

  for i in range(0, 10): 

    print("*************************************")  
    print("PROBLEM {}".format(i))
    
    s0 = PROBLEMS[i] 

    se = SearchEngine('breadth_first', 'full')
    se.init_search(s0, goal_fn=sokoban_goal_state, heur_fn=heur_displaced)
    final = se.search(timebound)

    if final:
      final.print_path()
      solved += 1
    else:
      unsolved.append(i)    
    counter += 1

  if counter > 0:  
    percent = (solved/counter)*100

  print("*************************************")  
  print("{} of {} problems ({} %) solved in less than {} seconds.".format(solved, counter, percent, timebound))  
  print("Problems that remain unsolved in the set are Problems: {}".format(unsolved))      
  print("*************************************") 
  print("*************************************") 
  print("*************************************") 
  print("*************************************") 

  solved = 0; unsolved = []; counter = 0; percent = 0; timebound = 2; #2 second time limit for each problem
  print("*************************************")  
  print("Running Best-first")     

  for i in range(0, 10):

    print("*************************************")  
    print("PROBLEM {}".format(i))
    
    s0 = PROBLEMS[i] 

    se = SearchEngine('best_first', 'full')
    se.init_search(s0, goal_fn=sokoban_goal_state, heur_fn=heur_displaced)
    final = se.search(timebound)

    if final:
      final.print_path()
      solved += 1
    else:
      unsolved.append(i)    
    counter += 1

  if counter > 0:  
    percent = (solved/counter)*100

  print("*************************************")  
  print("{} of {} problems ({} %) solved in less than {} seconds.".format(solved, counter, percent, timebound))  
  print("Problems that remain unsolved in the set are Problems: {}".format(unsolved))      
  print("*************************************") 
  print("*************************************") 
  print("*************************************") 
  print("*************************************") 

  solved = 0; unsolved = []; counter = 0; percent = 0; timebound = 2; #2 second time limit for each problem
  print("*************************************")  
  print("Running Uniform Cost Search")     

  for i in range(0, 10): 

    print("*************************************")  
    print("PROBLEM {}".format(i))
    
    s0 = PROBLEMS[i]

    se = SearchEngine('ucs', 'full')
    se.init_search(s0, goal_fn=sokoban_goal_state, heur_fn=heur_displaced)
    final = se.search(timebound)

    if final:
      final.print_path()
      solved += 1
    else:
      unsolved.append(i)    
    counter += 1

  if counter > 0:  
    percent = (solved/counter)*100

  print("*************************************")  
  print("{} of {} problems ({} %) solved in less than {} seconds.".format(solved, counter, percent, timebound))  
  print("Problems that remain unsolved in the set are Problems: {}".format(unsolved))      
  print("*************************************") 
  print("*************************************") 
  print("*************************************") 
  print("*************************************")