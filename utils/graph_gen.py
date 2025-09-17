"""
    Rohith kumar Senthil kumar
    09/04/2025
    Graph generation from image
"""

# --- imports ---
import utils.simple_graph_modified as gs
import math

class GraphGenerator:
    """Generates a graph from an image based on obstacles and free space."""
    def __init__(self, img=None, kernel_size=20, stride=20):
        self.img = img
        self.kernel_size = kernel_size
        self.stride = stride

    def check_if_obstacle(self, node):
        """check_if_obstacle: checks if the given node is an obstacle or not"""
        # Check if the node is an obstacle
        if node.sum() == 0:
            return True
        return False
    
    def find_blocks(self, x,y):
        """find_blocks: finds the blocks in which the given coordinates lie"""
        max_range_x = math.ceil(x/self.stride) * self.stride
        max_range_y = math.ceil(y/self.stride) * self.stride
        return max_range_y-self.stride, max_range_y, max_range_x-self.stride, max_range_x

    def block_position(self, id, total_columns):
        """block_position: finds the position of the block in the grid"""
        # if left or right boundary return an invalid id
        if id!=1 and id%total_columns==0:
            return "left"
        elif id!=1 and (id+1)%total_columns==0:
            return "right"
        else:
            return "center"
        
    def get_index_from_coordinates(self, coords):
        """get_index_from_coordinates: gets the index of the block from the coordinates"""
        w = self.img.shape[1]
        i, j = coords[0]//self.stride, coords[2]//self.stride
        return i * (w // self.stride) + j


    def set_neighbors(self, blocks, block, id, w):
        """set_neighbors: sets the neighbors of the given block"""

        # Initialize variables
        total_columns = w//self.stride
        top = blocks.get(id-total_columns, None)
        top_left = None
        top_right = None
        left = None
        right = None

        # Determine current block position
        current_block_position = self.block_position(id, total_columns)
        
        # Set neighbors based on position
        if current_block_position=="center":
            top_left = blocks.get(id-total_columns-1, None)
            top_right = blocks.get(id-total_columns+1, None)
            left = blocks.get(id-1, None)
            right = blocks.get(id+1, None)
        elif current_block_position=="left":
            top_right = blocks.get(id-total_columns+1, None)
        else:
            left = blocks.get(id-1, None)
            top_left = blocks.get(id-total_columns-1, None)

        # Add valid neighbors
        neighbors = [top_left, top, top_right, left, right]
        
        # Add neighbors to block
        for neighbor in neighbors:
            if neighbor is not None and not neighbor.isWall():
                block.addUndirectedNeighbor(neighbor) 

    def construct_graph(self, img):
        """construct_graph: constructs the graph from the image"""

        # Initialize variables
        id = 0
        blocks = {}
        kernel_size = self.kernel_size
        stride = self.stride
        w,h,c = img.shape
        
        # Construct graph
        try:
            for i in range(w//stride):
                for j in range(h//stride):
                    block = gs.Node(id=id, y1=stride * i, y2=kernel_size +  stride * i, x1=stride * j, x2=kernel_size + stride * j)
                    if self.check_if_obstacle(img[ stride * i: kernel_size +  stride * i,  stride * j: kernel_size + stride * j]):
                        block.setWall(True)
                    else:
                        block.setWall(False)
                        self.set_neighbors(blocks, block, id, h)
                    blocks[id] = block
                    id+=1
        except Exception as e:
            print(e)
            return False, blocks
        return True, blocks
    
    def execute(self):
        """execute: executes the graph generation"""
        return self.construct_graph(self.img)