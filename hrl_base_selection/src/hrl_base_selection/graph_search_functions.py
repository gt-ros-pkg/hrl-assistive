import collections
import heapq

class SimpleGraph:
    def __init__(self):
        self.edges = {}
        self.value = {}
    
    def neighbors(self, id):
        return self.edges[id]

    def value_of_node(self, id):
        return self.value[id]

class Queue:
    def __init__(self):
        self.elements = collections.deque()
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, x):
        self.elements.append(x)
    
    def get(self):
        return self.elements.popleft()

class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]

def reconstruct_path(came_from, start, goal):
    if goal not in came_from:
        # print 'The goal was never reached in the graph'
        return []
    current = goal
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    # path.append(start) # optional
    path.reverse() # optional
    return path

def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def a_star_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    value_so_far = {}
    came_from[start] = None
    value_so_far[start] = 0
    
    while not frontier.empty():
        current = frontier.get()
        
        if current == goal:
            break
        
        for next in graph.neighbors(current):
            new_value = value_so_far[current] + graph.value_of_node(next)
            if next not in value_so_far or new_value > value_so_far[next]:
                value_so_far[next] = new_value
                priority = new_value  # + heuristic(goal, next)
                frontier.put(next, priority)
                came_from[next] = current
    
    return came_from, value_so_far

if __name__ == "__main__":
    example_graph = SimpleGraph()
    example_graph.edges = {
        '00': ['A1', 'A2', 'A3', 'A4'],
        'A1': ['B1', 'B2'],
        'A2': ['B1', 'B2', 'B3'],
        'A3': ['B2', 'B3'],
        'A4': ['B1', 'B4'],
        'B1': [],
        'B2': ['C2', 'C4'],
        'B3': ['C1'],
        'B4': ['C3', 'C4'],
        'C1': ['D1'],
        'C2': ['D1'],
        'C3': [],
        'C4': ['D2'],
        'D1': ['Z0'],
        'D2': ['Z0'],
        'D3': ['Z0'],
        'D4': ['Z0']
    }
    example_graph.value = {
        '00': 0,
        'A1': 1,
        'A2': 2,
        'A3': 5,
        'A4': 1,
        'B1': 4,
        'B2': 3,
        'B3': 1,
        'B4': 2,
        'C1': 1,
        'C2': 2,
        'C3': 7,
        'C4': 2,
        'D1': 2,
        'D2': 3,
        'D3': 4,
        'D4': 5,
        'Z0': 0
    }

    example_graph2 = SimpleGraph()
    example_graph2.edges = {
        '0-0': ['1-0', '1-2'],
        '1-0': ['2-1'],
        '1-1': [],
        '1-2': [],
        '1-3': [],
        '2-0': [],
        '2-1': [],
        '2-2': [],
        '2-3': [],
        '3-0': [],
        '3-1': [],
        '3-2': [],
        '3-3': [],
        '4-0': [],
        '4-1': [],
        '4-2': [],
        '4-3': ['5-0'],
        '5-0': []
    }
    example_graph2.value = {
        '0-0': 0,
        '1-0': 1,
        '1-1': 2,
        '1-2': 5,
        '1-3': 1,
        '2-0': 4,
        '2-1': 3,
        '2-2': 1,
        '2-3': 2,
        '3-0': 1,
        '3-1': 2,
        '3-2': 7,
        '3-3': 2,
        '4-0': 2,
        '4-1': 3,
        '4-2': 4,
        '4-3': 5,
        '5-0': 0
    }

    # print example_graph.neighbors('00')

    came_from, value_so_far = a_star_search(example_graph2, '0-0', '5-0')
    # print came_from

    print value_so_far
    import numpy as np
    furthest_reached = np.argmax([t for t in ((int(a[0].split('-')[0]))
                                              for a in value_so_far.items())
                                  ])
    print furthest_reached
    print value_so_far.items()[furthest_reached]

    # print reconstruct_path(came_from, '0-0', 'Z0')

    # came_from, value_so_far = a_star_search(example_graph2, '00', 'Z0')

    # print came_from

    # print value_so_far

    # print reconstruct_path(came_from, '00', 'Z0')

    # print len(value_so_far)


