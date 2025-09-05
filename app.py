import cv2
import streamlit as st
import numpy as np
import time
from utils.algorithms_impl import AlgorithmsImpl
from utils.graph_gen import GraphGenerator
st.set_page_config(page_title="Obstacle Detection", layout="wide")
st.title("Path Matching")

if "algo_executed" not in st.session_state:
    st.session_state.algo_executed = False
    
if "execute" not in st.session_state:
    st.session_state.execute = False

if "img_size" not in st.session_state:
    st.session_state.img_size = (520, 1000, 3) 
    
if "search_coords" not in st.session_state:
    st.session_state.search_coords = []
    
if "target_coords" not in st.session_state:
    st.session_state.target_coords = set()
    
if "graph_generator" not in st.session_state:
    st.session_state.graph_generator = None
    
def reset_image():
    st.session_state.img = np.ones(st.session_state.img_size, dtype=np.uint8) * 255
    
def initialize_image():
    obstacle_1 = np.ones(st.session_state.img_size, dtype=np.uint8)*255
    # obstacle_1[100:120, 100:480] = 0
    obstacle_1[20:100, 20:80] = [0,0,0]
    obstacle_1[60:120, 120:200] = [0,0,0]
    obstacle_1[0:60, 240:400] = [0,0,0]
    obstacle_1[0:60, 240:400] = [0,0,0]
    obstacle_1[80:120, 240:280] = [0,0,0]
    obstacle_1[80:120, 240:280] = [0,0,0]
    obstacle_1[80:120, 300:380] = [0,0,0]
    obstacle_1[40:120, 420:600] = [0,0,0]
    obstacle_1[160:400, 140:160] = [0,0,0]


    obstacle_1[120:160, 0:40] = [0,0,0]
    obstacle_1[120:160, 80:600] = [0,0,0]
    obstacle_1[120:160, 640:1000] = [0,0,0]
    obstacle_1[120:160, 640:1000] = [0,0,0]

    obstacle_1[300:320, 600:620] = [0,0,255]
    obstacle_1[440:460, 20:40] = [0,0,255]
    st.session_state.img = obstacle_1.copy()


if "img" not in st.session_state:
    img_size=  st.session_state.img_size
    initialize_image()


def draw_grids(img):
    h, w, c = st.session_state.img_size
    # Number of grid divisions
    rows, cols = h//20, w//20  # 4 horizontal parts, 6 vertical parts

    # Draw horizontal lines
    for i in range(1, rows):
        y = i * h // rows
        cv2.line(img, (0, y), (w, y), (200, 200, 200), 1)  # green lines

    # Draw vertical lines
    for j in range(1, cols):
        x = j * w // cols
        cv2.line(img, (x, 0), (x, h), (200, 200, 200), 1)
    return img

if st.session_state.execute:
    print("Executing algorithm...")
    initialize_image()
    _graph_gen = GraphGenerator(img=st.session_state.img, kernel_size=20, stride=20)
    ret, blocks = _graph_gen.execute()
    print("ret: ", ret, len(blocks))

    target_coords = set()
    source_coords = [(0, 20, 0, 20)]
    target_coords.add((300, 320, 600, 620))
    target_coords.add((440, 460, 20, 40))
    st.session_state.target_coords = target_coords
    for coords in source_coords:
        source_index = _graph_gen.get_index_from_coordinates(coords)
        blocks[source_index].setIsSource(True)

    for coords in target_coords:
        target_index = _graph_gen.get_index_from_coordinates(coords)
        blocks[target_index].setIsTarget(True)
    st.session_state.graph_generator = _graph_gen

    print(blocks[0].isSource(), blocks[780].isTarget())

    match st.session_state.algorithm:
        case "Breadth-First Search":
            graph = list(blocks.values())
            res, st.session_state.search_coords = AlgorithmsImpl().bfs(graph[0], target_coords = target_coords.copy())
            print("bfs: ", res, "len", len(st.session_state.search_coords))
        case "Dijkstra":
            graph = list(blocks.values())
            res, st.session_state.search_coords = AlgorithmsImpl().a_star(graph, target_coords = target_coords.copy(), include_heuristic=False)
            print("dijkstra: ", res, "len", len(st.session_state.search_coords))
        case "A*":
            graph = list(blocks.values())
            res, st.session_state.search_coords = AlgorithmsImpl().a_star(graph, target_coords = target_coords.copy(), heuristic = st.session_state.heuristic.lower())
    print(f"{st.session_state.algorithm}: ", res, "len", len(st.session_state.search_coords))

col1, col2 = st.columns([5,2], vertical_alignment="top")
with col2:
    with st.container():
        st.selectbox("Select an algorithm", ["Breadth-First Search", "Dijkstra", "A*"], key="algorithm")
        if st.session_state.algorithm == "A*":
            st.selectbox("Heuristic", ["Manhattan", "Euclidean"], key="heuristic")
        st.button("Run" , on_click=lambda: st.session_state.update({"execute": True}))

placeholder = col1.empty()
if not st.session_state.execute:
    print('Algo not executed')
    # draw_grids()
    placeholder.image(draw_grids(st.session_state.img.copy()), caption='Obstacle 1')

else:
    print('Algo executed')
    for coord in st.session_state.search_coords:
        y1, y2, x1,x2 = coord
        st.session_state.img[y1:y2, x1:x2] = [127, 127, 127]
        
        st.markdown(
    """
    <style>
    img {
        -webkit-user-drag: none;
        user-drag: none;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
        placeholder.image(draw_grids(st.session_state.img.copy()), caption='Obstacle 1')
        time.sleep(0.02)

    for coords in list(st.session_state.target_coords):
        index = st.session_state.graph_generator.get_index_from_coordinates(coords)
        print(index)
        node = graph[index]
        while node.parent()!=-1 and node.parent() is not None:
            print(node.parent().id())
            node = node.parent()
            y1, y2, x1,x2 = node.get_coordinates()
            st.session_state.img[y1:y2, x1:x2] = [255, 255, 0]
    placeholder.image(draw_grids(st.session_state.img.copy()), caption='Obstacle 1')
    st.session_state.execute = False

