"""
    Rohith kumar Senthil kumar
    09/04/2025
    UI Utility functions
"""

# --- imports ---
import numpy as np
import cv2
from utils.graph_gen import GraphGenerator
from utils.algorithms_impl import AlgorithmsImpl
import streamlit as st

# --- main UIUtils class ---
class UIUtils:
    """Utility class for UI related functions."""
    def __init__(self, session_state):
        self.session_state = session_state

    def initialize_image(self):
        """initialize_image: initializes the image with default obstacle map"""
        obstacle_1 = np.ones(self.session_state.img_size, dtype=np.uint8) * 255
        if self.session_state.obstacle_map == "Sample":
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
        self.session_state.img = obstacle_1.copy()
        self.draw_source_and_targets(self.session_state.img)

    def draw_source_and_targets(self, img, draw_source=True, draw_target=True, draw_obstacle=True):
        """draw_source_and_targets: draws the source and target blocks on the image"""
        s = self.session_state
        if draw_source:
            for y1, y2, x1, x2 in s.click_coords["sources"]:
                img[y1:y2, x1:x2] = [255, 0, 0]
        if draw_target:
            for y1, y2, x1, x2 in s.click_coords["targets"]:
                img[y1:y2, x1:x2] = [0, 0, 255]
        if draw_obstacle:
            for y1, y2, x1, x2 in s.click_coords["obstacles"]:
                img[y1:y2, x1:x2] = [0, 0, 0]

    def draw_grids(self, img):
        """draw_grids: draws grid lines on the image"""
        h, w, _ = self.session_state.img_size
        rows, cols = h // 20, w // 20
        for i in range(1, rows):
            cv2.line(img, (0, i * h // rows), (w, i * h // rows), (200, 200, 200), 1)
        for j in range(1, cols):
            cv2.line(img, (j * w // cols, 0), (j * w // cols, h), (200, 200, 200), 1)
        return img

    def generate_graphs(self):
        """generate_graphs: generates graphs for each source and target"""
        
        # Initialize GraphGenerator
        _graph_gen = GraphGenerator(img=self.session_state.img, kernel_size=20, stride=20)
        
        # Generate graph, source coords, target coords
        generated_graphs, source_coords, target_coords = [], [], []
        sources = list(self.session_state.click_coords["sources"])
        targets = list(self.session_state.click_coords["targets"])

        # Generate graph for each source and target
        for coords in sources:
            ret, blocks = _graph_gen.execute()
            if ret:
                generated_graphs.append(blocks)
            source_coords.append(coords)
        for coords in targets:
            target_coords.append(coords)

        # Mark source and target nodes in the graphs
        for i, coords in enumerate(source_coords):
            idx = _graph_gen.get_index_from_coordinates(coords)
            generated_graphs[i][idx].setIsSource(True)
            for t in target_coords:
                tidx = _graph_gen.get_index_from_coordinates(t)
                generated_graphs[i][tidx].setIsTarget(True)
        return _graph_gen, generated_graphs, source_coords, target_coords

    def construct_frames(self, source_coords, target_coords, generated_graphs):
        """construct_frames: constructs frames for animation"""
        colors = [[77, 168, 218], [128, 216, 195], [255, 214, 107]]
        frames = []
        img = self.session_state.img.copy()
        
        # Draw initial grid with sources and targets
        for i, searchcoords in enumerate(self.session_state.search_coords_total):
            for y1, y2, x1, x2 in searchcoords:
                img[y1:y2, x1:x2] = colors[i % 3]
                frames.append(cv2.cvtColor(self.draw_grids(img.copy()), cv2.COLOR_BGR2RGB))
            self.draw_source_and_targets(img, draw_obstacle=False)
            frames.append(cv2.cvtColor(self.draw_grids(img.copy()), cv2.COLOR_BGR2RGB))

        # Draw paths for each source to its matched target
        for i, coords in enumerate(source_coords):
            
            graph = list(generated_graphs[i].values())
            target_index = self.session_state.matches.get(coords)
            if target_index is None:
                continue
            node = graph[self.session_state.graph_generator.get_index_from_coordinates(
                target_coords[target_index]
            )]
            while node.parent() not in (-1, None):
                node = node.parent()
                y1, y2, x1, x2 = node.get_coordinates()
                img[y1:y2, x1:x2] = [255, 255, 0]
            self.draw_source_and_targets(img, draw_obstacle=False)
            frames.append(cv2.cvtColor(self.draw_grids(img.copy()), cv2.COLOR_BGR2RGB))
        return frames

    def construct_quick_view_frame(self, sources_to_view):
        """construct_quick_view_frame: constructs a quick view frame showing selected paths"""
        frame = self.session_state.img.copy()
        
        # Draw paths for selected sources
        for source_idx in sources_to_view:
            # source_idx is 1-based index
            coords = self.session_state.source_coords[source_idx-1]
            graph = list(self.session_state.generated_graphs[source_idx-1].values())
            target_index = self.session_state.matches.get(coords)
            if target_index is None:
                continue
            node = graph[self.session_state.graph_generator.get_index_from_coordinates(
                self.session_state.target_coords[target_index]
            )]
            while node.parent() not in (-1, None):
                node = node.parent()
                y1, y2, x1, x2 = node.get_coordinates()
                frame[y1:y2, x1:x2] = [255, 255, 0]
            self.draw_source_and_targets(frame, draw_obstacle=False)
        return cv2.cvtColor(self.draw_grids(frame.copy()), cv2.COLOR_BGR2RGB)

    def validate_source_target(self):
        """validate_source_target: validates if at least one source and target is present"""
        if not self.session_state.click_coords["sources"]:
            self.session_state.status = "Please add at least one source block."
            self.session_state.execute = False
            return False
        if not self.session_state.click_coords["targets"]:
            self.session_state.status = "Please add at least one target block."
            self.session_state.execute = False
            return False
        self.session_state.status = "Ready to execute."
        return True
    
    def execute_algorithm(self):
        """execute_algorithm: executes the selected path matching algorithm"""
        if not self.validate_source_target():
            return
        self.initialize_image()
        _graph_gen, generated_graphs, source_coords, target_coords = self.generate_graphs()
        self.session_state.update(
            {
                "graph_generator": _graph_gen,
                "source_coords": source_coords,
                "generated_graphs": generated_graphs,
                "target_coords": target_coords,
            }
        )

        _, self.session_state.search_coords_total, self.session_state.matches, \
        self.session_state.shortest_path_time, self.session_state.bipartite_matching_time = (
            AlgorithmsImpl(
                generated_graphs,
                source_coords=source_coords.copy(),
                target_coords=target_coords.copy(),
                img_size=self.session_state.img_size,
                heuristic=self.session_state.heuristic,
            ).run(self.session_state.algorithm, self.session_state.matcher.lower())
        )

        self.session_state.execute = False
        self.session_state.configure_map = False
        self.session_state.frames = self.construct_frames(
            source_coords, target_coords, generated_graphs
        )
        self.session_state.allow_animation = True
        
    def reset(self, cache_data, cache_resource):
        """reset: resets the application state"""
        if "animator" in self.session_state:
            self.session_state.animator.stop_animation()
            self.session_state.animator.clear_frame_queue()
            del self.session_state["animator"]
        cache_data.clear()
        cache_resource.clear()
        st.session_state.clear()
        
    def add_block(self):
        """add_block: adds a block to the click coordinates based on the selected block type"""
        raw_value = self.session_state["click_coordinates"]
        x, y = raw_value["x"], raw_value["y"]
        block_coords = GraphGenerator(stride=20).find_blocks(x, y)
        key_map = {"🟥 Source": "sources", "🟦 Target": "targets", "⬛ Obstacle": "obstacles"}
        
        if block_coords in self.session_state.click_coords["sources"]:
            self.session_state.click_coords["sources"].remove(block_coords)
            self.session_state.total_sources -= 1
            if key_map[self.session_state.blocktype] == "sources":
                y1, y2, x1, x2 = block_coords
                self.session_state.img[y1:y2, x1:x2] = [255, 255, 255]
                return
        elif block_coords in self.session_state.click_coords["targets"]:
            self.session_state.click_coords["targets"].remove(block_coords)
            self.session_state.total_targets -= 1
            if key_map[self.session_state.blocktype] == "targets":
                y1, y2, x1, x2 = block_coords
                self.session_state.img[y1:y2, x1:x2] = [255, 255, 255]
                return
        elif block_coords in self.session_state.click_coords["obstacles"]:
            self.session_state.click_coords["obstacles"].remove(block_coords)
            self.session_state.total_obstacles -= 1
            if key_map[self.session_state.blocktype] == "obstacles":
                y1, y2, x1, x2 = block_coords
                self.session_state.img[y1:y2, x1:x2] = [255, 255, 255]
                return
        
        self.session_state.click_coords[key_map[self.session_state.blocktype]].add(block_coords)


