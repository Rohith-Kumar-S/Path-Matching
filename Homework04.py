"""
    Rohith kumar Senthil kumar
    06/04/2025
    CS5800 - Homework4
"""


# global variable for handling pre/post
step = 0


# Executes a DFS search on graph starting at node
def dfs_explore(graph, node, verbose=False):

    # use the global step variable
    global step
    # set the visited field of node to True
    node.setVisited(True)
    # set the pre field of node to step
    node.setPre(step)
    # increment step
    step += 1

    # for each neighbor n of node
        # if n has not been visited
            # set the parent of n to node
            # call dfs_explore with the graph, n, and verbose
    for n in node.neighbors():
            # if the node is not visited
            if not n.visited():
                # set the parent to p
                n.setParent(node)
                # append the node to the queue
                dfs_explore(graph, n, verbose=verbose)

    # set the post value to step
    node.setPost(step)
    # increment step
    step += 1

    if verbose:
        # print information about the node before returning
        parentid = -1
        if node.parent() != None:
            parentid = node.parent().id()
        print("Node %d (pre, post): (%d, %d)  parent %d" % (node.id(), node.pre(), node.post(), parentid))

    return


# Executes DFS over the whole graph
def dfs(graph, verbose=False):
    global step

    # set step to 0
    step = 0

    # initialize each node in the graph (visited -> False; pre -> -1; post -> -1; parent -> None)
    for node in graph:
        node.setVisited(False)
        node.setPre(-1)
        node.setPost(-1)
        node.setParent(None) 
        
    # for each node in the graph
        # if the node is not visited
            # call dfs_explore with graph, node, and verbose
    for node in graph:
        if not node.visited():
            dfs_explore(graph, node, verbose=True)

    return 


# Executes a BFS search starting at the given node
def bfs_explore(graph, node, verbose=False):
    global step
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
            if not node.visited() and node.val()!=0:
                # set visited to true
                node.setVisited(True)
                # set the parent to p
                node.setParent(p)
                # append the node to the queue
                Q.append(node)

    return


# Executes a BFS search of the entire graph
def bfs(graph, verbose=False):
    global step

    # initialize step to 0
    step = 0

    # initialize each node in the graph (visited -> False; pre -> -1; post -> -1; parent -> None)
    for node in graph:
        node.setVisited(False)
        node.setPre(-1)
        node.setPost(-1)
        node.setParent(None) 

    # for each node in the graph
        # if the node is not visited
            # call bfs_explore with graph, node, and verbose
    for node in graph:
        if not node.visited() and node.val()!=0:
            bfs_explore(graph, node, verbose=verbose)


# Test function for DFS and BFS search on a simple graph
def main():

    # build a graph
    print("1# Running DFS and BFS on a simple graph with 10 nodes")
    graph = []

    graph.append(gs.Node(0, 100, 100))
    graph.append(gs.Node(1, 200, 20))
    graph.append(gs.Node(2, 750, 100))
    graph.append(gs.Node(3, 550, 150))
    graph.append(gs.Node(4, 400, 250))
    graph.append(gs.Node(5, 850, 400))
    graph.append(gs.Node(6, 200, 350))
    graph.append(gs.Node(7, 550, 450))
    graph.append(gs.Node(8, 350, 550))
    graph.append(gs.Node(9, 120, 600))

    graph[0].addUndirectedNeighbor(graph[1])
    graph[0].addUndirectedNeighbor(graph[6])
    graph[1].addUndirectedNeighbor(graph[2])
    graph[1].addUndirectedNeighbor(graph[4])
    graph[2].addUndirectedNeighbor(graph[3])
    graph[2].addUndirectedNeighbor(graph[5])
    graph[3].addUndirectedNeighbor(graph[4])
    graph[3].addUndirectedNeighbor(graph[5])
    graph[3].addUndirectedNeighbor(graph[7])
    graph[4].addUndirectedNeighbor(graph[6])
    graph[6].addUndirectedNeighbor(graph[7])
    graph[6].addUndirectedNeighbor(graph[8])
    graph[6].addUndirectedNeighbor(graph[9])
    graph[7].addUndirectedNeighbor(graph[8])
    graph[8].addUndirectedNeighbor(graph[9])

    print("#1 Running DFS on the graph")
    dfs(graph, verbose=True)
    print("\n#1 Running BFS on the graph")
    bfs(graph, verbose=True)
    
    
    #2
    print("\n2# Running DFS and BFS on a simple graph with 14 nodes")
    graph = []
    graph.append(gs.Node(0, 100, 300))
    graph.append(gs.Node(1, 100, 100))
    graph.append(gs.Node(2, 100, 50))
    graph.append(gs.Node(3, 350, 300))
    graph.append(gs.Node(4, 350, 100))
    graph.append(gs.Node(5, 350, 50))
    graph.append(gs.Node(6, 500, 300))
    graph.append(gs.Node(7, 500, 100))
    graph.append(gs.Node(8, 700, 290))
    graph.append(gs.Node(9, 800, 100))
    graph.append(gs.Node(10, 800, 50))
    graph.append(gs.Node(11, 950, 100))
    graph.append(gs.Node(12, 950, 50))
    graph.append(gs.Node(13, 950, 0))
    
    graph[0].addUndirectedNeighbor(graph[1])
    graph[0].addUndirectedNeighbor(graph[3])
    graph[1].addUndirectedNeighbor(graph[2])
    graph[1].addUndirectedNeighbor(graph[4])
    graph[2].addUndirectedNeighbor(graph[5])
    graph[3].addUndirectedNeighbor(graph[6])
    graph[4].addUndirectedNeighbor(graph[7])
    graph[4].addUndirectedNeighbor(graph[5])
    graph[6].addUndirectedNeighbor(graph[4])
    graph[8].addUndirectedNeighbor(graph[9])
    graph[8].addUndirectedNeighbor(graph[10])
    graph[9].addUndirectedNeighbor(graph[11])
    graph[9].addUndirectedNeighbor(graph[10])
    graph[11].addUndirectedNeighbor(graph[12])
    graph[12].addUndirectedNeighbor(graph[13])
    
    for node in graph:
        s = "["
        for n in node.neighbors():
            s += str(n) + ", "
        s += "]"
        print(node, s)
    print("\n2# Running DFS on the graph")
    dfs(graph, verbose=True)
    print("\n2# Running BFS on the graph")
    bfs(graph, verbose=True)


    #3
    print("\n3# Running DFS on a random graph with varying vertex counts and densities")
    N = [20, 100, 500]
    percentages = [[], [], []]
    for i, vertex_count in enumerate(N):
        print("Running for vertex count:", vertex_count)
        for density in np.linspace(0, vertex_count, 25):
            graph = gt.buildRandomUndirectedGraph(vertex_count, density)
            dfs_explore(graph, graph[0], verbose=False)
            unvisited = 0
            for node in graph:
                if not node.visited():
                    unvisited += 1
            percentages[i].append(1-(unvisited/len(graph)))
    
    plt.plot(np.linspace(0, N[0], 25), percentages[0], label="vertices=20")
    plt.plot(np.linspace(0, N[1], 25), percentages[1], label="vertices=100")
    plt.plot(np.linspace(0, N[2], 25), percentages[2], label="vertices=500")
    plt.title("Average degree vs Percent of fully connected graphs")
    plt.xlabel("Average Degree")
    plt.ylabel("Percent of fully connected graphs")
    plt.xlim(0, 40)
    plt.grid()
    plt.legend(["vertices=20", "vertices=100", "vertices=500"])
    plt.show()

    return
    

if __name__ == "__main__":
    main()

