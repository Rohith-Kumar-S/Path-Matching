import heapq


class AlgorithmsImpl:
    
    def __init__(self):
        pass
    
    def get_index_from_coordinates(self, coords, stride, w):
        i, j = coords[0]//stride, coords[2]//stride
        return i * (w // stride) + j
    
    def euclidean_distance(self, y1, x1, y2, x2, stride):
        return ((((y2-y1)**2) + ((x2-x1)**2))**0.5)/stride
    
    def manhattan_distance(self, y1, x1, y2, x2, stride):
        return (abs(y2 - y1)/stride) + (abs(x2 - x1)/stride)
    
    def get_coordinate_midpoints(self, y1,y2, x1,x2):
        return (y1+y2)/2 , (x1+x2)/2
    
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
    
    def a_star(self, graphs, source_coords, target_coords, include_heuristic=True, heuristic = "euclidean"):
        match_found = False
        search_coords_total = []
        for i, coords in enumerate(source_coords):
            graph = list(graphs[i].values())
            source_block = graph[self.get_index_from_coordinates(coords, stride=20, w=1000)]
            match_found, search_coords = self.a_star_explore(graph, source_block, target_coords.copy(), include_heuristic=include_heuristic, heuristic=heuristic)
            print("match_found: ", match_found)
            search_coords_total.append(search_coords)
        return match_found, search_coords_total

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
    
    def bfs(self, graphs, source_coords, target_coords, verbose=False):
        search_coords_total = []
        match_found = False
        print(len(graphs), len(source_coords))
        for i, coords in enumerate(source_coords):
            graph = list(graphs[i].values())
            node = graph[self.get_index_from_coordinates(coords, stride=20, w=1000)]
            match_found, search_coords = self.bfs_explore(node, target_coords.copy(), verbose=verbose)
            print("match_found: ", match_found)
            search_coords_total.append(search_coords)
        return match_found, search_coords_total

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
