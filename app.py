import cv2
import streamlit as st
import numpy as np
import pandas as pd
from streamlit_image_coordinates import streamlit_image_coordinates

import asyncio
import threading
import time
from queue import Queue
from av import VideoFrame
from aiortc import VideoStreamTrack
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from utils.algorithms_impl import AlgorithmsImpl
from utils.graph_gen import GraphGenerator

st.set_page_config(page_title="Obstacle Detection", layout="wide")
st.title("Path Matching")

if "first_run" not in st.session_state:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.first_run = True

# Initialize required session_state keys
defaults = {
    "algo_executed": False,
    "execute": False,
    "img_size": (520, 1000, 3),
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
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
        
# Functions to initialize and draw on the image        

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


# --- webrtc streamer to display video from numpy array ---

# --- producer (same idea as yours) ---
frame_queue: "Queue[np.ndarray]" = Queue(maxsize=2)

col1, col2 = st.columns([5.5, 2], vertical_alignment="top")
statusholder = col2.empty()
statusholder.success("Enter Path Matching specifications and click Run")
def producer(frames):
    for img in frames:
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except:
                pass
        frame_queue.put(img)
        time.sleep(1/50)  # simulate ~60 fps

# --- custom aiortc VideoStreamTrack that yields frames from frame_queue ---
class NumpyVideoStreamTrack(VideoStreamTrack):
    def __init__(self, q: "Queue[np.ndarray]"):
        super().__init__()  # important
        self.q = q

    async def recv(self):
        # get next numpy frame from queue without blocking the event loop
        loop = asyncio.get_event_loop()
        img = await loop.run_in_executor(None, self.q.get)  # blocks off the event loop
        # convert to av.VideoFrame (bgr24 because OpenCV uses BGR)
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
            heuristic=st.session_state.heuristic,
        ).run(st.session_state.algorithm, st.session_state.matcher.lower())
    )

    st.session_state.algorithm_executed = True
    st.session_state.execute = False
    st.session_state.configure_map = False

    colors = [[77, 168, 218], [128, 216, 195], [255, 214, 107]]
    frames = []

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
    st.session_state.frames = frames
    st.session_state.algorithm_executed = False


if st.session_state.execute:
    execute_algorithm()
   
if "frames" in st.session_state and st.session_state.frames:
    threading.Thread(target=producer, args=(st.session_state.frames,), daemon=True).start()
st.markdown(
    """
    <style>
    img { -webkit-user-drag: none; user-drag: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

df = None
with col2:
    c1, c2 = st.columns(2, vertical_alignment="top")
    c1.selectbox("Select obstacle map", ["Sample", "Custom"],
                 key="obstacle_map", on_change=initialize_image)
    c2.selectbox("Insert Block", ["🟥 Source", "🟦 Target", "⬛ Obstacle"], key="blocktype")
    st.selectbox("Select an algorithm", ["Breadth-First Search", "Dijkstra", "A*"], key="algorithm")
    if st.session_state.algorithm == "A*":
        st.selectbox("Heuristic", ["Manhattan", "Euclidean"], key="heuristic")

    st.selectbox("Matcher", ["Greedy", "Linear Programming"], key="matcher")
    c3, c4, _ = st.columns([0.7, 1, 2], vertical_alignment="top")
    c3.button("Run", on_click=lambda: st.session_state.update({"execute": True}))
    if c4.button("Reset"):
        for k in ["frame_index", "shortest_path_time"]:
            st.session_state.pop(k, None)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.click_coords = {"sources": [], "targets": [], "obstacles": []}
        st.session_state.update({
            "execute": False,
            "configure_map": True,
            "algorithm_executed": False,
            "frames": [],
            "total_sources": 0,
            "total_targets": 0,
            "total_obstacles": 0,
        })
        initialize_image()
    if "frames" in st.session_state and st.session_state.frames:
        statusholder.success("Press Start to start simulation")

    if "shortest_path_time" in st.session_state and "bipartite_matching_time" in st.session_state:
        metrics = [f"Source {i}" for i, _ in enumerate(st.session_state.source_coords)]
        shortest_time = [f"{t*1000:.2f} ms" for t in st.session_state.shortest_path_time]
        df = pd.DataFrame({
            "Sources": metrics,
            "Shortest Path Time (ms)": shortest_time,
        }).set_index("Sources")

if df is not None:
    st.subheader("Results")
    st.write(df)
    st.text_input(
        "Bipartite Matching Time (ms)",
        value=f"{st.session_state.bipartite_matching_time*1000:.2f} ms",
    )

def add_block():
    raw_value = st.session_state["click_coordinates"]
    x, y = raw_value["x"], raw_value["y"]
    block_coords = GraphGenerator(stride=20).find_blocks(x, y)
    key_map = {"🟥 Source": "sources", "🟦 Target": "targets", "⬛ Obstacle": "obstacles"}
    st.session_state.click_coords[key_map[st.session_state.blocktype]].append(block_coords)

for k, (draw_args, total_key) in {
    "sources": ((True, False, False), "total_sources"),
    "targets": ((False, True, False), "total_targets"),
    "obstacles": ((False, False, True), "total_obstacles"),
}.items():
    if len(st.session_state.click_coords[k]) != st.session_state[total_key]:
        draw_source_and_targets(*draw_args)
        st.session_state[total_key] += 1

@st.cache_data
def fetch_frame(index):
    return st.session_state.frames[index]

with col1:
    if st.session_state.configure_map:
        streamlit_image_coordinates(
            draw_grids(st.session_state.img.copy()),
            key="click_coordinates",
            on_click=add_block,
        )
    # if "frames" in st.session_state and st.session_state.frames:
    if "frames" in st.session_state and st.session_state.frames:
        webrtc_ctx = webrtc_streamer(
            key="server-dummy-recvonly",
            mode=WebRtcMode.RECVONLY,       # IMPORTANT: browser will not call getUserMedia
            player_factory=player_factory, # returns object with .video attribute
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
        )
