import heapq
import numpy as np
import time
from scipy.optimize import linprog


class AlgorithmsImpl:
    
    def __init__(self, generated_graphs, source_coords, target_coords, img_size, heuristic=None):
        self.generated_graphs = generated_graphs
        self.source_coords = source_coords
        self.target_coords = target_coords
        self.heuristic = heuristic
        self.img_size = img_size
    
    def get_index_from_coordinates(self, coords, stride):
        w = self.img_size[1]
        i, j = coords[0]//stride, coords[2]//stride
        return i * (w // stride) + j
    
    def euclidean_distance(self, y1, x1, y2, x2, stride):
        return ((((y2-y1)**2) + ((x2-x1)**2))**0.5)/stride
    
    def manhattan_distance(self, y1, x1, y2, x2, stride):
        return (abs(y2 - y1)/stride) + (abs(x2 - x1)/stride)
    
    def get_coordinate_midpoints(self, y1,y2, x1,x2):
        return (y1+y2)/2 , (x1+x2)/2
    
    def greedy_weighted_bipartite_matching(self, edges):
        # edges = list of (u, v, w)
        flattened_edges = []
        for source_edges in edges:
            flattened_edges.extend(source_edges)
        flattened_edges.sort(key=lambda x: x[2])  # sort by weight
        matched_U = set()
        matched_V = set()
        matching = {}

        for u, v, w in flattened_edges:
            if u not in matched_U and v not in matched_V:
                matching[u] = v
                matched_U.add(u)
                matched_V.add(v)

        return matching
    
    def get_heruistic_cost(self, source_coords, target_coords, heruistic = "euclidean"):
        heruistic_method = None
        match heruistic:
            case "euclidean":
                heruistic_method = self.euclidean_distance
            case "manhattan":
                heruistic_method = self.manhattan_distance

        distances = []
        for coords in target_coords:
            distances.append(heruistic_method(
                *self.get_coordinate_midpoints(*source_coords),
                *self.get_coordinate_midpoints(*coords),
                stride=20
            ))
        return min(distances) if distances else float("inf")
    
    def build_linprog_params(self, sources, targets, distances):
        # Cost vector c (row-flattened)
        
        missing_indices = set()
        max_len = 0
        if len(sources) == len(targets):
            balance_condition = "balanced"
            max_len = len(sources)
        elif len(sources) < len(targets):
            balance_condition = "source_imbalance"
            max_len = len(targets)
        else:
            balance_condition = "target_imbalance"
            max_len = len(sources)

        if balance_condition == "target_imbalance":
            missing_targets = len(sources) - len(targets)
            for i in range(missing_targets):
                missing_indices.add(len(targets)+i)
                for distance_row in distances:
                    distance_row.append(('dummy_source','dummy_target', 1e6) * missing_targets)

        elif balance_condition == "source_imbalance":
            missing_sources = len(targets) - len(sources)
            for i in range(missing_sources):
                missing_indices.add(len(sources)+i)
                distances.append(('dummy_source','dummy_target', 1e6) * missing_sources)
        c = []
        for distances_row in distances:
            c.extend([d[-1] for d in distances_row])
        c = np.array(c)
        # Build constraints
        A_eq = []
        b_eq = []

        # Source constraints: each source assigned once
        for i in range(max_len):
            row = [0] * len(c)
            for j in range(max_len):
                row[i*max_len + j] = 1
            A_eq.append(row)
            b_eq.append(1)

        # Target constraints: each target assigned once
        for j in range(max_len):
            row = [0] * len(c)
            for i in range(max_len):
                row[i*max_len + j] = 1
            A_eq.append(row)
            b_eq.append(1)

        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)
        bounds = np.array([(0, 1)]*len(c))
        return c, A_eq, b_eq, bounds, sources, missing_indices, max_len

    def bipartite_linprog(self, c, A_eq, b_eq, bounds, sources, missing_indices, max_len):
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        
        matchings = {}
        
        for i in range(len(sources)**2):
            matched_edge = result.x[i]
            target_index = i%max_len
            if matched_edge > 0.5 and target_index not in missing_indices:
                matchings[sources[i//max_len]] = i%max_len
        return matchings
                
        
    def find_distance(self, graph, source_index, target_coords):
        distances = []
        for target_index, coords in enumerate(target_coords):
            distance = 0
            index = self.get_index_from_coordinates(coords, stride=20)
            node =  graph[index]
            while node.parent()!=-1 and node.parent() is not None:
                    node = node.parent()
                    distance+=1
            distances.append((source_index,target_index, distance))
        return distances
    
    def run(self, algorithm = "astar", matcher="greedy"):
        match_found = False
        search_coords_total = []
        distance_by_source = []
        matchings = None
        shortest_path_time = []
        bipartite_matching_time = 0
        for i, coords in enumerate(self.source_coords):
            graph = list(self.generated_graphs[i].values())
            source_block = graph[self.get_index_from_coordinates(coords, stride=20)]
            match algorithm:
                case "A*":
                    start_time = time.time()
                    match_found, search_coords = self.a_star_explore(graph, source_block, self.target_coords.copy(), heuristic=self.heuristic.lower())
                    shortest_path_time.append(time.time() - start_time)
                case "Dijkstra":
                    start_time = time.time()
                    match_found, search_coords = self.a_star_explore(graph, source_block, self.target_coords.copy(), include_heuristic=False)
                    shortest_path_time.append(time.time() - start_time)
                case "Breadth-First Search":
                    start_time = time.time()
                    match_found, search_coords = self.bfs_explore(source_block, self.target_coords.copy())
                    shortest_path_time.append(time.time() - start_time)
            distance_by_source.append(self.find_distance(graph, coords, self.target_coords))
            search_coords_total.append(search_coords)
        if matcher == "greedy":
            start_time = time.time()
            matchings = self.greedy_weighted_bipartite_matching(distance_by_source)
            bipartite_matching_time += time.time() - start_time
        else:
            start_time = time.time()
            matchings = self.bipartite_linprog(*self.build_linprog_params(self.source_coords, self.target_coords, distance_by_source))
            bipartite_matching_time += time.time() - start_time
        return match_found, search_coords_total, matchings, shortest_path_time, bipartite_matching_time

    def a_star_explore(self, graph, source_block, target_coords, include_heuristic=True, heuristic = "euclidean"):
        # target_coords = self.get_coordinate_midpoints(*target_coords)
        matches = 0
        blocks_to_match = len(target_coords)
        for node in graph:
            node.setParent(None)
            node.setCost(float("inf"))
            if include_heuristic:     
                node.setHeuristicCost(float("inf"))
            node.setVisited(False)
            
        source_block.setCost(0)
        if include_heuristic:
            source_block.setHeuristicCost(
                self.get_heruistic_cost(source_block.get_coordinates(), target_coords, heuristic)
            )
        queue = []
        heapq.heappush(queue, (source_block.cost() + source_block.getHeuristicCost() if include_heuristic else source_block.cost(), source_block.id(), source_block))  # f = g + h
        match_found = False
        search_coords = []
        while queue:
            _, id, current = heapq.heappop(queue)

            if current.visited():
                continue
            current.setVisited(True)

            # current_coords = self.get_coordinate_midpoints(*current.get_coordinates())
            current_coords = current.get_coordinates()
            # Goal check (if coords match target)
            if current.isTarget():
                matches += 1
                target_coords.remove(current_coords)
                if matches == blocks_to_match:
                    match_found = True
                    return True, search_coords

            for neighbor in current.neighbors():
                neighbor_coords = neighbor.get_coordinates()
                # neighbor_coords = self.get_coordinate_midpoints(*neighbor_block_coords)
                known_cost = current.cost() + self.get_heruistic_cost(current_coords, [neighbor_coords], heuristic)

                # known_cost = current.cost() + self.euclidean_distance(*current_coords, *neighbor_coords, stride=20)
                if not neighbor.visited() and known_cost < neighbor.cost():
                    search_coords.append((neighbor_coords))
                    neighbor.setCost(known_cost)
                    neighbor.setParent(current)
                    if include_heuristic:
                        heruistic_cost = self.get_heruistic_cost(neighbor_coords, target_coords, heuristic)
                        # heruistic_cost = self.euclidean_distance(*neighbor_coords, *target_coords, stride=20)
                        neighbor.setHeuristicCost(heruistic_cost)
                        new_cost = known_cost + heruistic_cost
                    else:
                        new_cost = known_cost
                        
                    heapq.heappush(queue, (new_cost, neighbor.id(), neighbor))
        return match_found, search_coords

    def bfs_explore(self, node, target_coords, verbose=False):
        matches = 0
        blocks_to_match = len(target_coords)
        step = 0
        match_found = False
        search_coords = []
        # set the node as visited
        node.setVisited(True)

        # initialize a queue (list)
        Q = []

        # append the node to the queue
        Q.append(node)

        # while the queue is not empty
        while Q:

            # take the first element p off the queue (pop(0))
            p = Q.pop(0)
            # set the pre value of p and increment step
            p.setPre(step)
            step += 1
            # set parentid to -1
            if p.parent() is None:
                p.setParent(-1)
            # if the parent node is not None
            if p.parent:
                # set parentid to the parent node ID
                parentid = p.id()
                
            if verbose:
                print("Node %d pre: %d parent %d" % (p.id(), p.pre(), parentid))
            # for each node in the neighbors list of p
            for node in p.neighbors():
                # if the node is not visited
                if node and not node.visited():
                    search_coords.append((node.get_coordinates()))
                    # set visited to true
                    node.setVisited(True)
                    # set the parent to p
                    node.setParent(p)
                    if node.isTarget():
                        print("Found target node %d" % (node.id()))
                        matches+=1
                        target_coords.remove(node.get_coordinates())
                        # If the target is found, we can stop the search
                        if matches == blocks_to_match:
                            match_found = True
                            return True, search_coords
                    # append the node to the queue
                    Q.append(node)

        return match_found, search_coords
