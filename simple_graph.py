# Bruce A. Maxwell
# Graph class
# Summer 2025
# HW #4

# a graph is a list of nodes
# each node is an object
# each node has a cost field
# each node has a visited field
# each node has a parent field
# each node has a neighbor field
#   the neighbor field is a list of adjacent nodes
# each node has an id field
# each node has an (x, y) field
# Class to represent a graph node, appropriate for DFS, BFS, Dijkstra's algorithm
class Node:

    # take in an x, y location and an id
    def __init__(self, id, x=0, y=0 ):
        self._x = x
        self._y = y
        self._id = id
        self._visited = False
        self._cost = 1000000
        self._parent = None
        self._pre = -1
        self._post = -1
        self._neighbors = []

    # accessor functions
    def id(self):
        return self._id
        
    def visited(self):
        return self._visited

    def setVisited(self, v):
        self._visited = v
        
    def cost(self):
        return self._cost

    def setCost(self, c):
        self._cost = c

    def parent(self):
        return self._parent

    def setParent(self, p):
        self._parent = p

    def pre(self):
        return self._pre

    def setPre(self, p):
        self._pre = p

    def post(self):
        return self._post

    def setPost(self, p):
        self._post = p

    def neighbors(self):
        return self._neighbors
    
    def get_coordinates(self):
        return (self._x, self._y)

    # adds a neighbor to this node's adjacency list, and this node to the neighbor's list
    def addUndirectedNeighbor( self, n ):
        if not (n in self._neighbors):
            self._neighbors.append(n)
            n._neighbors.append(self)

    # adds a neighbor to this node's adjacency list
    def addDirectedNeighbor( self, n ):
        if not (n in self._neighbors):
            self._neighbors.append(n)

    # compute the Euclidean distance between nodes
    def distance(self, n):
        dx = self._x - n._x
        dy = self._y - n._y

        return (dx*dx + dy*dy)**0.5

    # pretty printing
    def __str__(self):
        if self._parent != None:
            parent = self._parent._id
        else:
            parent = -1
        if self._visited:
            return "%d (%d, %d) %.1f -> %d" % (self._id, self._x, self._y, self._cost, parent)

        return "%d (%d, %d) inf -> %d" % (self._id, self._x, self._y, parent)



# Main test function
def main():

    # build a graph
    graph = []

    graph.append( Node(0, 100, 100) )
    graph.append( Node(1, 200,  20) )
    graph.append( Node(2, 750, 100) )
    graph.append( Node(3, 550, 150) )
    graph.append( Node(4, 400, 250) )
    graph.append( Node(5, 850, 400) )
    graph.append( Node(6, 200, 350) )
    graph.append( Node(7, 550, 450) )
    graph.append( Node(8, 350, 550) )
    graph.append( Node(9, 120, 600) )

    graph[0].addUndirectedNeighbor( graph[1] )
    graph[0].addUndirectedNeighbor( graph[6] )
    graph[1].addUndirectedNeighbor( graph[2] )
    graph[1].addUndirectedNeighbor( graph[4] )
    graph[2].addUndirectedNeighbor( graph[5] )
    graph[2].addUndirectedNeighbor( graph[3] )
    graph[3].addUndirectedNeighbor( graph[5] )
    graph[3].addUndirectedNeighbor( graph[4] )
    graph[3].addUndirectedNeighbor( graph[7] )
    graph[4].addUndirectedNeighbor( graph[6] )
    graph[6].addUndirectedNeighbor( graph[7] )
    graph[6].addUndirectedNeighbor( graph[8] )
    graph[6].addUndirectedNeighbor( graph[9] )
    graph[7].addUndirectedNeighbor( graph[8] )
    graph[8].addUndirectedNeighbor( graph[9] )

    #print the nodes in the graph and each adjacency list
    print("Graph:")
    for node in graph:
        s = "["
        for n in node.neighbors():
            s += str(n) + ", "
        s += "]"
        print(node, s)
        
    print(graph[2].pre())
            

if __name__ == "__main__":
    main()


                  
