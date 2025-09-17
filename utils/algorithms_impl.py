"""
    Rohith kumar Senthil kumar
    09/04/2025
    Algorithm implementations for path matching
"""

# --- imports ---
import heapq
import numpy as np
import time
from scipy.optimize import linprog

# --- main AlgorithmsImpl class ---
class AlgorithmsImpl:
    """Implements pathfinding and bipartite matching algorithms."""
    
    def __init__(self, generated_graphs, source_coords, target_coords, img_size, heuristic=None):
        self.generated_graphs = generated_graphs
        self.source_coords = source_coords
        self.target_coords = target_coords
        self.heuristic = heuristic
        self.img_size = img_size
    
    def get_index_from_coordinates(self, coords, stride):
        """Get graph index from image coordinates."""
        w = self.img_size[1]
        i, j = coords[0]//stride, coords[2]//stride
        return i * (w // stride) + j
    
    def euclidean_distance(self, y1, x1, y2, x2, stride):
        """Calculate Euclidean distance between two points, returned in grid units."""
        return ((((y2-y1)**2) + ((x2-x1)**2))**0.5)/stride
    
    def manhattan_distance(self, y1, x1, y2, x2, stride):
        """Calculate Manhattan distance between two points, returned in grid units."""
        return (abs(y2 - y1)/stride) + (abs(x2 - x1)/stride)
    
    def get_coordinate_midpoints(self, y1,y2, x1,x2):
        """Get midpoint of a coordinate block."""
        return (y1+y2)/2 , (x1+x2)/2
    
    def greedy_weighted_bipartite_matching(self, edges):
        """Find a maximum matching in a weighted bipartite graph using a greedy algorithm."""
        # edges = list of (u, v, w)
        flattened_edges = []
        
        # flatten the list of edges
        for source_edges in edges:
            flattened_edges.extend(source_edges)
        
        # sort by weight
        flattened_edges.sort(key=lambda x: x[2]) 
        matched_U = set()
        matched_V = set()
        matching = {}

        # Greedily add edges to the matching
        for u, v, w in flattened_edges:
            if u not in matched_U and v not in matched_V:
                matching[u] = v
                matched_U.add(u)
                matched_V.add(v)
                
        # return the final matching
        return matching

    def get_heuristic_cost(self, source_coords, target_coords, heuristic = "euclidean"):
        """Calculate heuristic cost from source to nearest target."""
        heuristic_method = None

        # Select heuristic method
        match heuristic:
            case "euclidean":
                heuristic_method = self.euclidean_distance
            case "manhattan":
                heuristic_method = self.manhattan_distance

        distances = []
        
        # Calculate distances to all targets and return the minimum
        for coords in target_coords:
            distances.append(heuristic_method(
                *self.get_coordinate_midpoints(*source_coords),
                *self.get_coordinate_midpoints(*coords),
                stride=20
            ))
        return min(distances) if distances else float("inf")
    
    def build_linprog_params(self, sources, targets, distances):
        """Build linear programming parameters for bipartite matching."""
        # Cost vector c (row-flattened)
        missing_indices = set()
        max_len = 0

        # Determine balance condition, sources vs targets
        if len(sources) == len(targets):
            balance_condition = "balanced"
            max_len = len(sources)
        elif len(sources) < len(targets):
            balance_condition = "source_imbalance"
            max_len = len(targets)
        else:
            balance_condition = "target_imbalance"
            max_len = len(sources)

        # handle imbalance by adding dummy nodes with high cost
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
                distances.append([('dummy_source','dummy_target', 1e6)] * len(targets))
        
        # build parameters
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
        return c, A_eq, b_eq, bounds, sources, missing_indices, max_len, balance_condition

    def bipartite_linprog(self, c, A_eq, b_eq, bounds, sources, missing_indices, max_len, balance_condition):
        """Solve bipartite matching using linear programming."""
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        
        matchings = {}
        
        # match results back to source-target pairs
        for i in range(max_len * max_len):
            matched_edge = result.x[i]
            target_index = None
            if balance_condition == "target_imbalance":
                target_index = i%max_len
            if matched_edge > 0.5 and target_index not in missing_indices and len(sources)>i//max_len:
                matchings[sources[i//max_len]] = i%max_len
        return matchings
                
        
    def find_distance(self, graph, source_index, target_coords):
        """Find distance from source to each target in the graph."""
        distances = []
        
        # for each target, trace back to source to find distance
        for target_idx, target_coords in enumerate(target_coords):
            distance = 0
            index = self.get_index_from_coordinates(target_coords, stride=20)
            node =  graph[index]
            while node.parent()!=-1 and node.parent() is not None:
                    node = node.parent()
                    distance+=1
            distances.append((source_index, target_idx, distance))
        return distances
    
    def run(self, algorithm = "astar", matcher="greedy"):
        """Run the selected pathfinding and matching algorithms."""
        
        # Initialize variables
        match_found = False
        search_coords_total = []
        distance_by_source = []
        matchings = None
        shortest_path_time = []
        bipartite_matching_time = 0
        
        # for each source, run the selected pathfinding algorithm
        for i, coords in enumerate(self.source_coords):
            graph = list(self.generated_graphs[i].values())
            source_block = graph[self.get_index_from_coordinates(coords, stride=20)]
            match algorithm:
                case "A*":
                    start_time = time.time()
                    match_found, search_coords = self.a_star_explore(graph, source_block, self.target_coords, heuristic=self.heuristic.lower())
                    shortest_path_time.append(time.time() - start_time)
                case "Dijkstra":
                    start_time = time.time()
                    match_found, search_coords = self.a_star_explore(graph, source_block, self.target_coords, include_heuristic=False)
                    shortest_path_time.append(time.time() - start_time)
                case "Breadth-First Search":
                    start_time = time.time()
                    match_found, search_coords = self.bfs_explore(source_block, self.target_coords)
                    shortest_path_time.append(time.time() - start_time)
            distance_by_source.append(self.find_distance(graph, coords, self.target_coords))
            search_coords_total.append(search_coords)
            
        # run the selected matching algorithm
        if matcher == "greedy":
            start_time = time.time()
            matchings = self.greedy_weighted_bipartite_matching(distance_by_source)
            bipartite_matching_time += time.time() - start_time
        else:
            start_time = time.time()
            matchings = self.bipartite_linprog(*self.build_linprog_params(self.source_coords, self.target_coords, distance_by_source))
            bipartite_matching_time += time.time() - start_time
        
        # return results
        return match_found, search_coords_total, matchings, shortest_path_time, bipartite_matching_time

    def a_star_explore(self, graph, source_block, target_coords, include_heuristic=True, heuristic = "euclidean"):
        """Perform A* search from source to targets in the graph."""
        
        # Initialize variables
        matches = 0
        blocks_to_match = len(target_coords)
        matched_targets = set()
        match_found = False
        search_coords = []
        queue = []
        
        # Initialize all nodes
        for node in graph:
            node.setParent(None)
            node.setCost(float("inf"))
            if include_heuristic:     
                node.setHeuristicCost(float("inf"))
            node.setVisited(False)
        
        # Initialize source node
        source_block.setCost(0)
        if include_heuristic:
            source_block.setHeuristicCost(
                self.get_heruistic_cost(source_block.get_coordinates(), target_coords, heuristic)
            )
        # Add source to priority queue
        heapq.heappush(queue, (source_block.cost() + source_block.getHeuristicCost() if include_heuristic else source_block.cost(), source_block.id(), source_block))  # f = g + h

        # A* search loop
        while queue:
            _, id, current = heapq.heappop(queue)

            if current.visited():
                continue
            current.setVisited(True)
            current_coords = current.get_coordinates()
            
            # Goal check (if coords match target)
            if current.isTarget() and current_coords not in matched_targets:
                matches += 1
                matched_targets.add(current_coords)
                if matches == blocks_to_match:
                    match_found = True
                    return True, search_coords

            for neighbor in current.neighbors():
                neighbor_coords = neighbor.get_coordinates()
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
                    
                    # add neighbor to priority queue
                    heapq.heappush(queue, (new_cost, neighbor.id(), neighbor))
        return match_found, search_coords

    def bfs_explore(self, node, target_coords, verbose=False):
        """Perform Breadth-First Search from source to targets in the graph."""
        
        # Initialize variables
        matches = 0
        blocks_to_match = len(target_coords)
        step = 0
        match_found = False
        search_coords = []
        matched_targets = set()
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
                    if node.isTarget() and node.get_coordinates() not in matched_targets:
                        print("Found target node %d" % (node.id()))
                        matches+=1
                        matched_targets.add(node.get_coordinates())
                        # If the target is found, we can stop the search
                        if matches == blocks_to_match:
                            match_found = True
                            return True, search_coords
                    # append the node to the queue
                    Q.append(node)

        return match_found, search_coords
