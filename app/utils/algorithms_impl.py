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
        match heruistic:
            case "euclidean":
                return self.euclidean_distance(
                    *self.get_coordinate_midpoints(*source_coords),
                    *self.get_coordinate_midpoints(*target_coords),
                    stride=20
                )
            case "manhattan":
                return self.manhattan_distance(
                    *self.get_coordinate_midpoints(*source_coords),
                    *self.get_coordinate_midpoints(*target_coords),
                    stride=20
                )

    def a_star(self, graph, target_coords, include_heuristic=True, heuristic = "euclidean"):
        # target_coords = self.get_coordinate_midpoints(*target_coords)
        for node in graph:
            node.setParent(None)
            node.setCost(float("inf"))
            if include_heuristic:     
                node.setHeuristicCost(float("inf"))
            node.setVisited(False)

        start = graph[0]
        start.setCost(0)
        if include_heuristic:
            print("heuristic: ", self.get_heruistic_cost(start.get_coordinates(), target_coords, heuristic), heuristic)
            start.setHeuristicCost(
                self.get_heruistic_cost(start.get_coordinates(), target_coords, heuristic)
                # self.euclidean_distance(
                #     *self.get_coordinate_midpoints(*start.get_coordinates()),
                #     *target_coords,
                #     stride=20
                # )
            )
        queue = []
        heapq.heappush(queue, (start.cost() + start.getHeuristicCost() if include_heuristic else start.cost(), start.id(), start))  # f = g + h
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
            if current_coords == target_coords:
                match_found = True
                break

            for neighbor in current.neighbors():
                neighbor_coords = neighbor.get_coordinates()
                # neighbor_coords = self.get_coordinate_midpoints(*neighbor_block_coords)
                known_cost = current.cost() + self.get_heruistic_cost(current_coords, neighbor_coords, heuristic)

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

    def bfs(self, node, verbose=False):
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
                        # If the target is found, we can stop the search
                        return True, search_coords
                    # append the node to the queue
                    Q.append(node)

        return match_found, search_coords
