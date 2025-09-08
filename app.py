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
   
if "source_coords" not in st.session_state:
    st.session_state.source_coords = []

if "search_coords" not in st.session_state:
    st.session_state.search_coords = []
    
if "target_coords" not in st.session_state:
    st.session_state.target_coords = set()
    
if "graph_generator" not in st.session_state:
    st.session_state.graph_generator = None

if "generated_graphs" not in st.session_state:
    st.session_state.generated_graphs = []
    
if "heuristic" not in st.session_state:
    st.session_state.heuristic = 'Manhattan'
    
if "matches" not in st.session_state:
    st.session_state.matches = None
    
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

    obstacle_1[0:20, 0:20] = [255,0,0]
    obstacle_1[180:200, 380:400] = [255,0,0]
    obstacle_1[500:520, 980:1000] = [255,0,0]
    obstacle_1[300:320, 600:620] = [0,0,255]
    obstacle_1[440:460, 20:40] = [0,0,255]
    st.session_state.img = obstacle_1.copy()
    
def draw_source_and_targets():
    st.session_state.img[0:20, 0:20] = [255,0,0]
    st.session_state.img[180:200, 380:400] = [255,0,0]
    st.session_state.img[500:520, 980:1000] = [255,0,0]
    st.session_state.img[300:320, 600:620] = [0,0,255]
    st.session_state.img[440:460, 20:40] = [0,0,255]
    


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
    generated_graphs = []
    for i in range(3):
        ret, blocks = _graph_gen.execute()
        if ret:
            generated_graphs.append(blocks)
    print("ret: ", ret, len(generated_graphs))
    st.session_state.generated_graphs = generated_graphs
    source_coords = []
    target_coords = set()
    source_coords.append((0, 20, 0, 20))
    source_coords.append((180, 200, 380, 400))
    source_coords.append((500, 520, 980, 1000))
    target_coords.add((300, 320, 600, 620))
    target_coords.add((440, 460, 20, 40))
    st.session_state.target_coords = target_coords
    for i, coords in enumerate(source_coords):
        source_index = _graph_gen.get_index_from_coordinates(coords)
        generated_graphs[i][source_index].setIsSource(True)

    for i, _ in enumerate(source_coords):
        for coords in target_coords:
            target_index = _graph_gen.get_index_from_coordinates(coords)
            generated_graphs[i][target_index].setIsTarget(True)
    st.session_state.graph_generator = _graph_gen
    st.session_state.source_coords = source_coords
    print(blocks[0].isSource(), blocks[780].isTarget())
    res, st.session_state.search_coords_total, st.session_state.matches = AlgorithmsImpl(st.session_state.generated_graphs, source_coords = source_coords.copy(), target_coords = target_coords.copy(), heuristic = st.session_state.heuristic).run(st.session_state.algorithm)
    print(f"{st.session_state.algorithm}: ", res, "len", len(st.session_state.search_coords_total))

col1, col2 = st.columns([5,2], vertical_alignment="top")
with col2:
    with st.container():
        st.selectbox("Select an algorithm", ["Breadth-First Search", "Dijkstra", "A*"], key="algorithm")
        if st.session_state.algorithm == "A*":
            st.selectbox("Heuristic", ["Manhattan", "Euclidean"], key="heuristic")
        st.button("Run" , on_click=lambda: st.session_state.update({"execute": True}))

placeholder = col1.empty()
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
if not st.session_state.execute:
    print('Algo not executed')
    # draw_grids()
    placeholder.image(draw_grids(st.session_state.img.copy()), caption='Obstacle 1')

else:
    print('Algo executed')
    colors = [[77, 168, 218], [128, 216, 195], [255, 214, 107]]
    for i, searchcoords in enumerate(st.session_state.search_coords_total):
        graph = list(st.session_state.generated_graphs[i].values())
        for coords in searchcoords:
            y1, y2, x1,x2 = coords
            st.session_state.img[y1:y2, x1:x2] = colors[i]
            placeholder.image(draw_grids(st.session_state.img.copy()), caption='Obstacle 1')
            time.sleep(0.01)
        draw_source_and_targets()
    for i, coords in enumerate(st.session_state.source_coords):
        graph = list(st.session_state.generated_graphs[i].values())
        target_index = st.session_state.matches.get(coords, None)
        if target_index is None:
            continue 
        index = st.session_state.graph_generator.get_index_from_coordinates(list(st.session_state.target_coords)[target_index])
        node =  graph[index]
        while node.parent()!=-1 and node.parent() is not None:
            node = node.parent()
            y1, y2, x1,x2 = node.get_coordinates()
            st.session_state.img[y1:y2, x1:x2] = [255, 255, 0]
        draw_source_and_targets()
        placeholder.image(draw_grids(st.session_state.img.copy()), caption='Obstacle 1')
    st.session_state.execute = False
