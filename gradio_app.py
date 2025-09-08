import cv2
import gradio as gr
import numpy as np
import time
from typing import Dict, List, Tuple, Set
# These imports assume you have the same utility modules
from utils.algorithms_impl import AlgorithmsImpl
from utils.graph_gen import GraphGenerator


class PathMatchingApp:
    def __init__(self):
        self.img_size = (520, 1000, 3)
        self.initialize_state()
    
    def initialize_state(self):
        """Initialize or reset all state variables"""
        self.algo_executed = False
        self.execute = False
        self.source_coords = []
        self.search_coords = []
        self.search_coords_total = []
        self.target_coords = set()
        self.graph_generator = None
        self.generated_graphs = []
        self.heuristic = 'Manhattan'
        self.matches = None
        self.algorithm = "Breadth-First Search"
        self.initialize_image()
    
    def reset_image(self):
        """Reset image to blank white canvas"""
        self.img = np.ones(self.img_size, dtype=np.uint8) * 255
    
    def initialize_image(self):
        """Initialize image with obstacles and source/target points"""
        obstacle_1 = np.ones(self.img_size, dtype=np.uint8) * 255
        
        # Draw obstacles (black areas)
        obstacle_1[20:100, 20:80] = [0, 0, 0]
        obstacle_1[60:120, 120:200] = [0, 0, 0]
        obstacle_1[0:60, 240:400] = [0, 0, 0]
        obstacle_1[80:120, 240:280] = [0, 0, 0]
        obstacle_1[80:120, 300:380] = [0, 0, 0]
        obstacle_1[40:120, 420:600] = [0, 0, 0]
        obstacle_1[160:400, 140:160] = [0, 0, 0]
        obstacle_1[120:160, 0:40] = [0, 0, 0]
        obstacle_1[120:160, 80:600] = [0, 0, 0]
        obstacle_1[120:160, 640:1000] = [0, 0, 0]
        
        # Draw source points (red)
        obstacle_1[0:20, 0:20] = [255, 0, 0]
        obstacle_1[180:200, 380:400] = [255, 0, 0]
        obstacle_1[500:520, 980:1000] = [255, 0, 0]
        
        # Draw target points (blue)
        obstacle_1[300:320, 600:620] = [0, 0, 255]
        obstacle_1[440:460, 20:40] = [0, 0, 255]
        
        self.img = obstacle_1.copy()
        self.original_img = obstacle_1.copy()
    
    def draw_source_and_targets(self):
        """Redraw source and target points on the image"""
        self.img[0:20, 0:20] = [255, 0, 0]
        self.img[180:200, 380:400] = [255, 0, 0]
        self.img[500:520, 980:1000] = [255, 0, 0]
        self.img[300:320, 600:620] = [0, 0, 255]
        self.img[440:460, 20:40] = [0, 0, 255]
    
    def draw_grids(self, img):
        """Draw grid lines on the image"""
        h, w, c = self.img_size
        img_copy = img.copy()
        
        # Number of grid divisions
        rows, cols = h // 20, w // 20
        
        # Draw horizontal lines
        for i in range(1, rows):
            y = i * h // rows
            cv2.line(img_copy, (0, y), (w, y), (200, 200, 200), 1)
        
        # Draw vertical lines
        for j in range(1, cols):
            x = j * w // cols
            cv2.line(img_copy, (x, 0), (x, h), (200, 200, 200), 1)
        
        return img_copy
    
    def run_algorithm_with_animation(self, algorithm, heuristic, animation_speed):
        """Execute the selected pathfinding algorithm with animation"""
        self.algorithm = algorithm
        self.heuristic = heuristic if algorithm == "A*" else "Manhattan"
        
        # Reset to initial state
        self.initialize_image()
        
        # Yield initial state
        yield self.draw_grids(self.img.copy()), "Initializing graph..."
        
        # Generate graphs
        self.graph_generator = GraphGenerator(img=self.img, kernel_size=20, stride=20)
        self.generated_graphs = []
        
        for i in range(3):
            ret, blocks = self.graph_generator.execute()
            if ret:
                self.generated_graphs.append(blocks)
        
        # Set up source and target coordinates
        self.source_coords = [
            (0, 20, 0, 20),
            (180, 200, 380, 400),
            (500, 520, 980, 1000)
        ]
        
        self.target_coords = {
            (300, 320, 600, 620),
            (440, 460, 20, 40)
        }
        
        # Mark sources and targets in the graph
        for i, coords in enumerate(self.source_coords):
            source_index = self.graph_generator.get_index_from_coordinates(coords)
            self.generated_graphs[i][source_index].setIsSource(True)
        
        for i, _ in enumerate(self.source_coords):
            for coords in self.target_coords:
                target_index = self.graph_generator.get_index_from_coordinates(coords)
                self.generated_graphs[i][target_index].setIsTarget(True)
        
        yield self.draw_grids(self.img.copy()), f"Running {algorithm}..."
        
        # Run the algorithm
        algo_impl = AlgorithmsImpl(
            self.generated_graphs,
            source_coords=self.source_coords.copy(),
            target_coords=self.target_coords.copy(),
            heuristic=self.heuristic
        )
        
        res, self.search_coords_total, self.matches = algo_impl.run(algorithm)
        
        # Visualize the search process with animation
        colors = [[77, 168, 218], [128, 216, 195], [255, 214, 107]]
        
        # Calculate sleep time based on animation speed
        sleep_time = (100 - animation_speed) / 1000.0  # Convert to seconds
        
        # Animate search process
        for i, searchcoords in enumerate(self.search_coords_total):
            graph = list(self.generated_graphs[i].values())
            for j, coords in enumerate(searchcoords):
                y1, y2, x1, x2 = coords
                self.img[y1:y2, x1:x2] = colors[i]
                
                # Yield every few steps to show animation
                if j % max(1, int(10 - animation_speed/10)) == 0:
                    self.draw_source_and_targets()
                    yield self.draw_grids(self.img.copy()), f"Searching from source {i+1}/3..."
                    time.sleep(sleep_time)
            
            self.draw_source_and_targets()
            yield self.draw_grids(self.img.copy()), f"Completed search from source {i+1}/3"
        
        yield self.draw_grids(self.img.copy()), "Drawing final paths..."
        
        # Draw final paths
        for i, coords in enumerate(self.source_coords):
            graph = list(self.generated_graphs[i].values())
            target_index = self.matches.get(coords, None)
            if target_index is None:
                continue
            
            index = self.graph_generator.get_index_from_coordinates(
                list(self.target_coords)[target_index]
            )
            node = graph[index]
            
            path_nodes = []
            while node.parent() != -1 and node.parent() is not None:
                node = node.parent()
                path_nodes.append(node.get_coordinates())
            
            # Animate path drawing
            for coords in path_nodes:
                y1, y2, x1, x2 = coords
                self.img[y1:y2, x1:x2] = [255, 255, 0]
                self.draw_source_and_targets()
                yield self.draw_grids(self.img.copy()), f"Drawing path {i+1}/3..."
                time.sleep(sleep_time)
        
        self.draw_source_and_targets()
        yield self.draw_grids(self.img.copy()), f"✓ {algorithm} complete! Matches found: {len(self.matches)}"
    
    def create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(title="Path Matching", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🔍 Path Matching Visualization")
            gr.Markdown("Watch pathfinding algorithms (BFS, Dijkstra, A*) explore a grid with obstacles in real-time.")
            
            with gr.Row():
                with gr.Column(scale=5):
                    # Image display
                    image_output = gr.Image(
                        value=self.draw_grids(self.img.copy()),
                        label="Obstacle Map",
                        type="numpy",
                        height=520,
                        elem_id="main-image"
                    )
                    
                    # Status bar
                    status = gr.Textbox(
                        label="Status",
                        value="Ready to run algorithm",
                        interactive=False,
                        elem_id="status-bar"
                    )
                
                with gr.Column(scale=2):
                    # Algorithm selection
                    algorithm_dropdown = gr.Dropdown(
                        choices=["Breadth-First Search", "Dijkstra", "A*"],
                        value="Breadth-First Search",
                        label="Select Algorithm",
                        info="Choose the pathfinding algorithm to visualize"
                    )
                    
                    # Heuristic selection (only visible for A*)
                    heuristic_dropdown = gr.Dropdown(
                        choices=["Manhattan", "Euclidean"],
                        value="Manhattan",
                        label="Heuristic Function",
                        info="Distance calculation method for A*",
                        visible=False
                    )
                    
                    # Animation speed slider
                    speed_slider = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Animation Speed",
                        info="Control how fast the search animation plays"
                    )
                    
                    # Control buttons
                    with gr.Row():
                        run_button = gr.Button("▶️ Run", variant="primary", scale=2)
                        reset_button = gr.Button("🔄 Reset", variant="secondary", scale=1)
                    
                    # Legend
                    with gr.Accordion("📊 Legend", open=True):
                        gr.Markdown("""
                        | Color | Meaning |
                        |-------|---------|
                        | 🟥 **Red** | Source points |
                        | 🟦 **Blue** | Target points |
                        | ⬛ **Black** | Obstacles |
                        | 🟨 **Yellow** | Final path |
                        | 🔵 **Light Blue** | Search from source 1 |
                        | 🟢 **Light Green** | Search from source 2 |
                        | 🟡 **Light Yellow** | Search from source 3 |
                        """)
                    
                    # Info panel
                    with gr.Accordion("ℹ️ About", open=False):
                        gr.Markdown("""
                        This visualization demonstrates how different pathfinding algorithms 
                        explore the search space to find optimal paths from multiple sources 
                        to targets while avoiding obstacles.
                        
                        - **BFS**: Explores uniformly in all directions
                        - **Dijkstra**: Finds shortest path using edge weights
                        - **A***: Uses heuristic to guide search toward target
                        """)
            
            # Show/hide heuristic dropdown based on algorithm selection
            def update_heuristic_visibility(algorithm):
                return gr.update(visible=(algorithm == "A*"))
            
            algorithm_dropdown.change(
                fn=update_heuristic_visibility,
                inputs=[algorithm_dropdown],
                outputs=[heuristic_dropdown]
            )
            
            # Reset function
            def reset_visualization():
                self.initialize_image()
                return self.draw_grids(self.img.copy()), "Ready to run algorithm"
            
            reset_button.click(
                fn=reset_visualization,
                inputs=[],
                outputs=[image_output, status]
            )
            
            # Run algorithm with animation
            run_button.click(
                fn=self.run_algorithm_with_animation,
                inputs=[algorithm_dropdown, heuristic_dropdown, speed_slider],
                outputs=[image_output, status]
            )
            
            # Add custom CSS for better styling
            interface.css = """
            #main-image {
                border: 2px solid #e0e0e0;
                border-radius: 8px;
            }
            #status-bar {
                font-family: monospace;
                background-color: #f5f5f5;
            }
            """
        
        return interface


# Create and launch the app
if __name__ == "__main__":
    app = PathMatchingApp()
    interface = app.create_interface()
    interface.launch(share=True)