
import utils.simple_graph_modified as gs

class GraphGenerator:
    
    def __init__(self, img, kernel_size, stride):
        self.img = img
        self.kernel_size = kernel_size
        self.stride = stride

    def check_if_obstacle(self, node):
        # Check if the node is an obstacle
        if node.sum() == 0:
            return True
        return False

    def block_position(self, id, total_columns):
        # if left or right boundary return an invalid id
        if id!=1 and id%total_columns==0:
            return "left"
        elif id!=1 and (id+1)%total_columns==0:
            return "right"
        else:
            return "center"
        
    def get_index_from_coordinates(self, coords):
        w = self.img.shape[1]
        i, j = coords[0]//self.stride, coords[2]//self.stride
        return i * (w // self.stride) + j


    def set_neighbors(self, blocks, block, id, w):
        total_columns = w//self.stride
        top = blocks.get(id-total_columns, None)
        top_left = None
        top_right = None
        left = None
        right = None
        current_block_position = self.block_position(id, total_columns)
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
            
        neighbors = [top_left, top, top_right, left, right]

        for neighbor in neighbors:
            if neighbor is not None and not neighbor.isWall():
                block.addUndirectedNeighbor(neighbor)
                

    def construct_graph(self, img):
        id = 0
        blocks = {}
        kernel_size = self.kernel_size
        stride = self.stride
        w,h,c = img.shape
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
        return self.construct_graph(self.img)