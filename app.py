import cv2
import streamlit as st
import numpy as np
import time
import pandas as pd
from streamlit_image_coordinates import streamlit_image_coordinates
import asyncio
import threading
import time
from queue import Queue
from av import VideoFrame
from aiortc import VideoStreamTrack
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import atexit
from utils.algorithms_impl import AlgorithmsImpl
from utils.graph_gen import GraphGenerator
import logging

# 1️⃣ Configure logging once, near the top of your script
logging.basicConfig(
    level=logging.INFO,                 # or DEBUG for more detail
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
st.set_page_config(page_title="Obstacle Detection", layout="wide")
st.title("Path Matching")
if "first_run" not in st.session_state:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.first_run = True
    st.session_state.animation_thread = None

# Initialize required session_state keys
defaults = {
    "algo_executed": False,
    "execute": False,
    "img_size": (500, 500, 3),
    "source_coords": [],
    "search_coords": [],
    "target_coords": set(),
    "graph_generator": None,
    "generated_graphs": [],
    "heuristic": "Manhattan",
    "matches": None,
    "click_coords": {"sources": [], "targets": [], "obstacles": []},
    "total_sources": 0,
    "total_targets": 0,
    "total_obstacles": 0,
    "algorithm_executed": False,
    "configure_map": True,
    "obstacle_map": "Sample",
    "frames": [],
    "view_mode": "Quick",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
    
if "view_mode" in st.session_state and st.session_state.view_mode=="Animated":
    stop_event = threading.Event()
    
def initialize_image():
    obstacle_1 = np.ones(st.session_state.img_size, dtype=np.uint8) * 255
    if st.session_state.obstacle_map == "Sample":
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
    st.session_state.img = obstacle_1.copy()
    draw_source_and_targets()

def draw_source_and_targets(draw_source=True, draw_target=True, draw_obstacle=True):
    s = st.session_state
    if draw_source:
        for y1, y2, x1, x2 in s.click_coords["sources"]:
            s.img[y1:y2, x1:x2] = [255, 0, 0]
    if draw_target:
        for y1, y2, x1, x2 in s.click_coords["targets"]:
            s.img[y1:y2, x1:x2] = [0, 0, 255]
    if draw_obstacle:
        for y1, y2, x1, x2 in s.click_coords["obstacles"]:
            s.img[y1:y2, x1:x2] = [0, 0, 0]

if "img" not in st.session_state:
    initialize_image()

def draw_grids(img):
    h, w, _ = st.session_state.img_size
    rows, cols = h // 20, w // 20
    for i in range(1, rows):
        cv2.line(img, (0, i * h // rows), (w, i * h // rows), (200, 200, 200), 1)
    for j in range(1, cols):
        cv2.line(img, (j * w // cols, 0), (j * w // cols, h), (200, 200, 200), 1)
    return img

# If Animation is enabled
frame_queue: "Queue[np.ndarray]" = Queue(maxsize=2)
def producer(frames):
    """Producer function that runs through frames once and stops"""
    for img in frames:
        # Check if we should stop early
        if stop_event.is_set():
            break
            
        # Remove old frame if queue is full
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except:
                pass
                
        # Add frame to queue
        frame_queue.put(img)
        time.sleep(1/50) 
    # Keep the last frame displayed
    if frames and not stop_event.is_set():
        # Ensure the last frame stays in the queue
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except:
                pass
        frame_queue.put(frames[-1])

# --- custom aiortc VideoStreamTrack that yields frames from frame_queue ---
class NumpyVideoStreamTrack(VideoStreamTrack):
    def __init__(self, q: "Queue[np.ndarray]"):
        super().__init__()  # important
        self.q = q
        self.last_frame = None

    async def recv(self):
        # get next numpy frame from queue without blocking the event loop

        loop = asyncio.get_event_loop()
        try:
            img = await asyncio.wait_for(
                loop.run_in_executor(None, self.q.get, True, 0.1),
                timeout=0.5
            )
            self.last_frame = img
        except (asyncio.TimeoutError, Exception):
            # Use last frame if available, otherwise create blank frame
            if self.last_frame is not None:
                img = self.last_frame
            else:
                img = np.ones(st.session_state.img_size, dtype=np.uint8) * 255
        frame = VideoFrame.from_ndarray(img, format="bgr24")

        # set pts/time_base for correct timing
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        return frame

# --- wrapper object streamlit-webrtc expects: has .video attribute ---
class PlayerLike:
    def __init__(self, video_track):
        self.video = video_track
        self.audio = None

def player_factory():
    # return the wrapper that contains our video track
    return PlayerLike(NumpyVideoStreamTrack(frame_queue))



def execute_algorithm():
    # put inside a class
    if not st.session_state.click_coords["sources"]:
        statusholder.warning("Please add at least one source block.")
        st.session_state.execute = False
        return
    if not st.session_state.click_coords["targets"]:
        statusholder.warning("Please add at least one target block.")
        st.session_state.execute = False
        return
    initialize_image()
    _graph_gen = GraphGenerator(img=st.session_state.img, kernel_size=20, stride=20)
    generated_graphs, source_coords, target_coords = [], [], set()

    for coords in st.session_state.click_coords["sources"]:
        ret, blocks = _graph_gen.execute()
        if ret:
            generated_graphs.append(blocks)
        source_coords.append(coords)
    for coords in st.session_state.click_coords["targets"]:
        target_coords.add(coords)

    for i, coords in enumerate(source_coords):
        idx = _graph_gen.get_index_from_coordinates(coords)
        generated_graphs[i][idx].setIsSource(True)
        for t in target_coords:
            tidx = _graph_gen.get_index_from_coordinates(t)
            generated_graphs[i][tidx].setIsTarget(True)

    st.session_state.update(
        {
            "graph_generator": _graph_gen,
            "source_coords": source_coords,
            "generated_graphs": generated_graphs,
            "target_coords": target_coords,
        }
    )

    res, st.session_state.search_coords_total, st.session_state.matches, \
    st.session_state.shortest_path_time, st.session_state.bipartite_matching_time = (
        AlgorithmsImpl(
            generated_graphs,
            source_coords=source_coords.copy(),
            target_coords=target_coords.copy(),
            img_size = st.session_state.img_size,
            heuristic=st.session_state.heuristic,
        ).run(st.session_state.algorithm, st.session_state.matcher.lower())
    )

    st.session_state.algorithm_executed = True
    st.session_state.execute = False
    st.session_state.configure_map = False

    colors = [[77, 168, 218], [128, 216, 195], [255, 214, 107]]
    frames = []
    logger.info("Starting to append frames for visualization")
    logger.debug("Starting to append frames for visualization")
    for i, searchcoords in enumerate(st.session_state.search_coords_total):
        for y1, y2, x1, x2 in searchcoords:
            st.session_state.img[y1:y2, x1:x2] = colors[i % 3]
            frames.append(cv2.cvtColor(draw_grids(st.session_state.img.copy()), cv2.COLOR_BGR2RGB))
        draw_source_and_targets(draw_obstacle=False)
        frames.append(cv2.cvtColor(draw_grids(st.session_state.img.copy()), cv2.COLOR_BGR2RGB))

    for i, coords in enumerate(source_coords):
        graph = list(generated_graphs[i].values())
        target_index = st.session_state.matches.get(coords)
        if target_index is None:
            continue
        node = graph[st.session_state.graph_generator.get_index_from_coordinates(
            list(target_coords)[target_index]
        )]
        while node.parent() not in (-1, None):
            node = node.parent()
            y1, y2, x1, x2 = node.get_coordinates()
            st.session_state.img[y1:y2, x1:x2] = [255, 255, 0]
        draw_source_and_targets(draw_obstacle=False)
        frames.append(cv2.cvtColor(draw_grids(st.session_state.img.copy()), cv2.COLOR_BGR2RGB))
    logger.info("Action finished successfully")
    logger.debug("Action finished successfully")
    st.session_state.frames = frames
    st.session_state.algorithm_executed = False


if st.session_state.execute:
    execute_algorithm()

if  "view_mode" in st.session_state and st.session_state.view_mode=="Animated":
    if "animation_thread" in st.session_state and st.session_state.animation_thread:
        stop_event.set()
        if st.session_state.animation_thread.is_alive():
            st.session_state.animation_thread.join(timeout=0.5)
        stop_event.clear()
    if len(st.session_state.frames)>0:
        st.session_state.animation_thread = threading.Thread(target=producer, args=(st.session_state.frames,), daemon=True)
        st.session_state.animation_thread.start()
    
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
col1, col2 = st.columns([5.5,2], vertical_alignment="top")
df = None
with col2:
    with st.container():
        view_mode_placeholder = st.empty()
        slider_placeholder = st.empty()
        c1,c2 = st.columns(2, vertical_alignment="top")
        c1.selectbox("Select obstacle map", [ "Sample", "Custom"], key="obstacle_map", on_change=initialize_image)
        c2.selectbox("Insert Block", ["🟥 Source", "🟦 Target", "⬛ Obstacle"], key="blocktype")
        st.selectbox("Select an algorithm", ["Breadth-First Search", "Dijkstra", "A*"], key="algorithm")
        if st.session_state.algorithm == "A*":
            st.selectbox("Heuristic", ["Manhattan", "Euclidean"], key="heuristic")
        
        st.selectbox("Matcher", ["Greedy", "Linear Programming"], key="matcher")
        
        c3, c4, c5 = st.columns([0.7,1, 2], vertical_alignment="top")
        c3.button("Run" , on_click=lambda: st.session_state.update({"execute": True}))
        if c4.button("Reset"):
            if "frame_index" in st.session_state:
                del st.session_state["frame_index"]
            if "shortest_path_time" in st.session_state:
                del st.session_state["shortest_path_time"]
            if "view_mode" in st.session_state and st.session_state.view_mode=="Animated":
                stop_event.clear()
                # Clear the frame queue
                while not frame_queue.empty():
                    try:
                        frame_queue.get_nowait()
                    except:
                        break
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.click_coords = {'sources': [], 'targets': [], 'obstacles': []}
            initialize_image()
            st.session_state.update({
                "execute": False,
                "configure_map": True,
                "algorithm_executed": False,
                "frames": [],
                "total_sources": 0,
                "total_targets": 0,
                "total_obstacles": 0
            })
            initialize_image()
        options = ["Quick", "Animated"]
        view_mode_placeholder.segmented_control(
                "View Mode",  options=options, selection_mode="single", key="view_mode"
            )
        if len(st.session_state.frames) > 0 and st.session_state.view_mode=="Quick":
            st.session_state.frame_index = slider_placeholder.slider(
            "Select frame", 0, len(st.session_state.frames) - 1, 0
        )
        if "shortest_path_time" in st.session_state and "bipartite_matching_time" in st.session_state:
            metrics = []
            shortest_time = []
            for i, source in enumerate(st.session_state.source_coords):
               metrics.append(f"Source {i}")
               shortest_time.append(f"{st.session_state.shortest_path_time[i]*1000:.2f} ms")
            df = pd.DataFrame({
                "Sources": metrics,
                "Shortest Path Time (ms)": shortest_time
            })
            df.set_index("Sources", inplace=True)
statusholder = st.empty()
statusholder.success("Enter Path Matching specifications and click Run")
if "frames" in st.session_state and st.session_state.frames:
    statusholder.success("Press Start to start simulation")
if df is not None:
    st.subheader("Results")
    st.write(df)
    st.text_input("Bipartite Matching Time (ms)", value=f"{st.session_state.bipartite_matching_time*1000:.2f} ms")
            
def add_block():
    raw_value = st.session_state["click_coordinates"]
    x, y = raw_value["x"], raw_value["y"]
    block_coords = GraphGenerator(stride=20).find_blocks(x, y)
    key_map = {"🟥 Source": "sources", "🟦 Target": "targets", "⬛ Obstacle": "obstacles"}
    st.session_state.click_coords[key_map[st.session_state.blocktype]].append(block_coords)

if len(st.session_state.click_coords['sources'])!= st.session_state.total_sources:
    draw_source_and_targets(draw_source=True, draw_target=False)
    st.session_state.total_sources+=1
if len(st.session_state.click_coords['targets'])!= st.session_state.total_targets:
    draw_source_and_targets(draw_source=False, draw_target=True)
    st.session_state.total_targets+=1
if len(st.session_state.click_coords['obstacles'])!= st.session_state.total_obstacles:
    draw_source_and_targets(draw_source=False, draw_target=False, draw_obstacle=True)
    st.session_state.total_obstacles+=1

@st.cache_data
def fetch_frame(index):
    return st.session_state.frames[index]
with col1:
    if st.session_state.configure_map:
        coords = streamlit_image_coordinates(draw_grids(st.session_state.img.copy()), key="click_coordinates", on_click=add_block)

    if len(st.session_state.frames) > 0 and "view_mode" in st.session_state:
        if st.session_state.view_mode=="Quick":
            st.image(cv2.cvtColor(fetch_frame(st.session_state.frame_index), cv2.COLOR_BGR2RGB))
        elif st.session_state.view_mode=="Animated":
            webrtc_ctx = webrtc_streamer(
            key="server-dummy-recvonly",
            mode=WebRtcMode.RECVONLY,       # IMPORTANT: browser will not call getUserMedia
            player_factory=player_factory, # returns object with .video attribute
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
        )

if st.session_state.animation_thread is not None and st.session_state.view_mode=="Animated":
    stop_event.set()
    st.session_state.animation_thread.join()
