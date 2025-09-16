import asyncio
import time
from queue import Queue
import numpy as np
from av import VideoFrame
from aiortc import VideoStreamTrack
import threading


class AnimationComponent:
    def __init__(self, stop_event, img_size):
        self.frame_queue: "Queue[np.ndarray]" = Queue(maxsize=2)
        self.stop_event = stop_event
        self.img_size = img_size
        self.animation_thread = None
        
    def get_frame_queue(self):
        return self.frame_queue
    
    def clear_frame_queue(self):
        self.stop_event.clear()
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
    
    def start_animation(self, frames=None):
        
        # Stop any existing animation thread
        self.stop_animation()
        if self.animation_thread is None:
            self.animation_thread = threading.Thread(target=self.producer, args=(frames,), daemon=True)
            self.animation_thread.start()
            
    def stop_animation(self):
        if self.animation_thread is not None:
            self.stop_event.set()
            if self.animation_thread.is_alive():
                self.animation_thread.join(timeout=0.5)
            self.animation_thread = None

    def player_factory(self):
        # return the wrapper that contains our video track
        return PlayerLike(NumpyVideoStreamTrack(self.frame_queue))
    def producer(self, frames): 
        """Producer function that runs through frames once and stops"""
        for img in frames:
            # Check if we should stop early
            if self.stop_event.is_set():
                break
                
            # Remove old frame if queue is full
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
                    
            # Add frame to queue
            self.frame_queue.put(img)
            time.sleep(1/50) 
        # Keep the last frame displayed
        if frames and not self.stop_event.is_set():
            # Ensure the last frame stays in the queue
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
            self.frame_queue.put(frames[-1])

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
                img = np.ones(self.img_size, dtype=np.uint8) * 255
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


