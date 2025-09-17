"""
    Rohith kumar Senthil kumar
    09/04/2025
    Path Matching Visualization Tool
"""

# --- imports ---
import cv2
import streamlit as st
import pandas as pd
from streamlit_image_coordinates import streamlit_image_coordinates
import threading
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from utils.animation_component import AnimationComponent
from utils.ui_utils import UIUtils
import time

# --- main app ---
st.set_page_config(page_title="Path Matching", layout="wide")
st.title("Path Matching")

# Setup cache and session state
if "first_run" not in st.session_state:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.first_run = True
    st.session_state.animation_thread = None

# Initialize required session_state keys
defaults = {
    "algo_executed": False,
    "execute": False,
    "img_size": (500, 600, 3),
    "source_coords": [],
    "search_coords": [],
    "target_coords": set(),
    "graph_generator": None,
    "generated_graphs": [],
    "heuristic": "Manhattan",
    "matches": None,
    "click_coords": {"sources": set(), "targets": set(), "obstacles": set()},
    "total_sources": 0,
    "total_targets": 0,
    "total_obstacles": 0,
    "algorithm_executed": False,
    "configure_map": True,
    "obstacle_map": "Sample",
    "frames": [],
    "quick_view_frame":[],
    "view_mode": "Quick",
    "animation_running": False,
    "allow_animation": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "stop_event" not in st.session_state:
    st.session_state.stop_event = threading.Event()
    
# Layout setup
col1, col2, col3 = st.columns([1.8,2.5,2], vertical_alignment="top")
df = None

# Initialize UIUtils
uiutils = UIUtils(
    session_state=st.session_state
)

if "img" not in st.session_state:
    uiutils.initialize_image()

# if animation is enabled, initialize animator
if "animator" not in st.session_state:
    st.session_state.animator =  AnimationComponent(st.session_state.stop_event, st.session_state.img_size)

# Execute the algorithm if requested
if st.session_state.execute:
    try:
        uiutils.execute_algorithm()
        if st.session_state.animation_running:
            st.session_state.animator.stop_animation()
            st.session_state.animator.clear_frame_queue()
            st.session_state.animation_running = False  
    except MemoryError:
        st.toast("Too Many Sources/Targets to process", icon="⚠️")
        time.sleep(2)
        uiutils.reset(st.cache_data, st.cache_resource, st.session_state.animator)
    except Exception as e:
        st.toast(f"Error: {e}", icon="⚠️")
        time.sleep(2)
        uiutils.reset(st.cache_data, st.cache_resource, st.session_state.animator)

if  "view_mode" in st.session_state and st.session_state.view_mode=="Animated" and len(st.session_state.frames)>0:
    if not st.session_state.animation_running and st.session_state.allow_animation:
        st.session_state.animator.start_animation(st.session_state.frames)
        st.session_state.animation_running = True
        st.session_state.allow_animation = False
elif st.session_state.animation_running:
    st.session_state.animator.stop_animation()
    st.session_state.animator.clear_frame_queue()
    st.session_state.animation_running = False

# Disable image dragging
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

# Sidebar for controls
with col1:
    with st.container():
        options = ["Quick", "Slider", "Animated"]
        st.segmented_control(
                "View Mode",  options=options, selection_mode="single", key="view_mode"
            )
        slider_placeholder = st.empty()
        c1,c2 = st.columns(2, vertical_alignment="top")
        c1.selectbox("Select obstacle map", [ "Sample", "Custom"], key="obstacle_map", on_change=uiutils.initialize_image)
        c2.selectbox("Insert Block", ["🟥 Source", "🟦 Target", "⬛ Obstacle"], key="blocktype")
        st.selectbox("Select an algorithm", ["Breadth-First Search", "Dijkstra", "A*"], key="algorithm")
        if st.session_state.algorithm == "A*":
            st.selectbox("Heuristic", ["Manhattan", "Euclidean"], key="heuristic")
        
        st.selectbox("Matcher", ["Greedy", "Linear Programming"], key="matcher")
        
        c3, c4, c5 = st.columns([0.7,1, 2.6], vertical_alignment="top")
        c3.button("Run" , on_click=lambda: st.session_state.update({"execute": True}))
        # c4.button("Reset", on_click=uiutils.reset, args=(st.cache_data, st.cache_resource))
        if len(st.session_state.frames) > 0 and st.session_state.view_mode=="Slider":
            st.session_state.frame_index = slider_placeholder.slider(
            "Select frame", 0, len(st.session_state.frames) - 1, 0
        )
        if "shortest_path_time" in st.session_state and "bipartite_matching_time" in st.session_state:
            metrics = []
            shortest_time = []
            path_matched = []
            view_path = []
            for i, source in enumerate(st.session_state.source_coords):
               metrics.append(i+1)
               shortest_time.append(f"{st.session_state.shortest_path_time[i]*1000:.2f} ms")
               view_path.append(True)
               path_matched.append(True)
            df = pd.DataFrame({
                "Sources": metrics,
                "Shortest Path Time (ms)": shortest_time,
                "Path Matched": path_matched,
                "View Path": view_path
            })

# Display status and results
statusholder = col3.empty()
if "status" in st.session_state and st.session_state.status.startswith("Please"):
    statusholder.warning(st.session_state.status)
elif "frames" in st.session_state and st.session_state.frames:
    if st.session_state.view_mode=="Animated":
        statusholder.success("Experimental Feature(Unstable) \n\n Press Run to start animation")
    elif st.session_state.view_mode=="Slider":
        statusholder.success("Use the slider to view different frames")
    else:
        statusholder.success("Click on the checkboxes in the results table to view paths")
else:
    statusholder.success("Insert blocks and click Run\n\n 🟥 Source 🟦 Target ⬛ Obstacle")

if df is not None:
    col3.subheader("Results")
    edited_df = col3.data_editor(df,disabled=["Sources", "Shortest Path Time (ms)", "Path Matched"],
    hide_index=True,)
    sources_to_view = edited_df.loc[edited_df["View Path"]==True]["Sources"].values
    if st.session_state.view_mode=="Quick" and len(st.session_state.frames)>0:
        st.session_state.quick_view_frame = uiutils.construct_quick_view_frame(sources_to_view)

    col3.text_input("Bipartite Matching Time (ms)", value=f"{st.session_state.bipartite_matching_time*1000:.2f} ms")

# Image display and interaction
if len(st.session_state.click_coords['sources'])!= st.session_state.total_sources:
    uiutils.draw_source_and_targets(st.session_state.img, draw_source=True, draw_target=False)
    st.session_state.total_sources+=1
if len(st.session_state.click_coords['targets'])!= st.session_state.total_targets:
    uiutils.draw_source_and_targets(st.session_state.img, draw_source=False, draw_target=True)
    st.session_state.total_targets+=1
if len(st.session_state.click_coords['obstacles'])!= st.session_state.total_obstacles:
    uiutils.draw_source_and_targets(st.session_state.img, draw_source=False, draw_target=False, draw_obstacle=True)
    st.session_state.total_obstacles+=1

@st.cache_data
def fetch_frame(index):
    return st.session_state.frames[index]

# Image display and interaction
with col2:
    if st.session_state.configure_map:
        coords = streamlit_image_coordinates(uiutils.draw_grids(st.session_state.img.copy()), key="click_coordinates", on_click=uiutils.add_block)

    if len(st.session_state.frames) > 0 and "view_mode" in st.session_state:
        if st.session_state.view_mode=="Slider":
            if st.session_state.animation_running:
                st.session_state.animator.stop_animation()
                st.session_state.animator.clear_frame_queue()
                st.session_state.animation_running = False
            st.image(cv2.cvtColor(fetch_frame(st.session_state.frame_index), cv2.COLOR_BGR2RGB))
        elif st.session_state.view_mode=="Animated":
            webrtc_ctx = webrtc_streamer(
            key="server-dummy-recvonly",
            mode=WebRtcMode.RECVONLY,
            player_factory=st.session_state.animator.player_factory,
            desired_playing_state=st.session_state.animation_running,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
        )
        elif st.session_state.view_mode=="Quick": # Quick view
            if st.session_state.animation_running:
                st.session_state.animator.stop_animation()
                st.session_state.animator.clear_frame_queue()
                st.session_state.animation_running = False
            st.image(cv2.cvtColor(st.session_state.quick_view_frame, cv2.COLOR_BGR2RGB))
# st.session_state.execute = False
# Stop animation thread if view mode changed or app is reset
if st.session_state.animation_thread is not None and st.session_state.view_mode=="Animated":
    st.session_state.animator.stop_animation()
