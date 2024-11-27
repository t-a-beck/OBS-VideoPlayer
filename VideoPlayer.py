import time

import cv2
import threading
from queue import Queue
import numpy as np
import platform

from frame_processing.ProcessFrame import ProcessFrame

DEFAULT_VID_PATH = "test/Open_af_Alter-Rally!.mp4"  # Set to 0 for default webcam
HEIGHT = 1080
WIDTH = 1920



class VideoPlayer:
	def mouse_callback(self, event, x, y, flags, param):

		"""Handle mouse events for zooming. Note only works for Windows so MAC workaround with clicking"""
		zoom_speed = 1.0
		max_zoom = 10.0
		if event == cv2.EVENT_MOUSEWHEEL and platform.system() == "Windows":
			if flags > 0:  # Scroll up
				self.zoom_factor = min(self.zoom_factor + zoom_speed, max_zoom)  # Max zoom factor
			elif flags < 0:  # Scroll down
				self.zoom_factor = max(self.zoom_factor - zoom_speed, 1.0)  # Min zoom factor

		elif event == cv2.EVENT_FLAG_LBUTTON or event == cv2.EVENT_FLAG_RBUTTON:
			if event == 1:  # Left click
				self.zoom_factor = min(self.zoom_factor + zoom_speed, max_zoom)  # Max zoom factor
			# print("zooming in ...")
			elif event == 2:  # Right click
				self.zoom_factor = max(self.zoom_factor - zoom_speed, 1.0)  # Min zoom factor
			print(f"Zoom factor: {self.zoom_factor}")


		elif event == cv2.EVENT_MOUSEMOVE:

			self.zoom_center = (x, y)

	def __init__(self, buffer_size: int = 100, frame_processor: ProcessFrame = ProcessFrame(),
				 reset_processor_on_load: bool = True):
		# Initialize playback window and basic variables
		cv2.namedWindow('Video Player')
		self.cap = None
		self.video_path = None
		self.video_fps = 0
		self.paused = False
		self.playback_speed = 30  # in milliseconds
		self.reset_processor_on_load = reset_processor_on_load
		self.cur_time_stamp = 0
		self.get_frame_at_timestamp = False
		self.frame_delivered = False

		# Buffering of frames
		self.buffer_size = buffer_size
		self.frame_buffer = Queue(maxsize=self.buffer_size)
		self.stop_buffer_thread = False
		self.buffer_thread = None
		self.lock = threading.Lock()

		# Replay status
		self.replay_active = False
		self.post_process = False
		self.save_frame = None

		#Zooming
		self.zoom_factor = 1.0
		self.zoom_center = (WIDTH // 2, HEIGHT // 2)  # Absolute center (x, y)
		cv2.setMouseCallback('Video Player', self.mouse_callback)

		# Processing Function
		self.frame_processor = frame_processor

	def load_video(self, video_path: str = DEFAULT_VID_PATH):
		"""Load the video file and initialize playback."""
		self.video_path = video_path
		self.cap = cv2.VideoCapture(video_path)
		if not self.cap.isOpened():
			raise ValueError("Error: Cannot open the video file.")
		self.paused = False

		self.time_add_per_frame = 1000/self.cap.get(cv2.CAP_PROP_FPS)
		# Start buffering
		self.stop_buffer_thread = False
		self.buffer_thread = threading.Thread(target=self.frame_reader)
		self.buffer_thread.daemon = True
		self.buffer_thread.start()

		if self.reset_processor_on_load:
			self.frame_processor = self.frame_processor.__class__()

	def set_replay_status(self, status: bool = False, post_process: bool = False) -> None:
		"""External programs call this to update replay status."""
		with self.lock:
			self.replay_active = status
			print(f"Replay status set to: {self.replay_active}")
			self.post_process = post_process
			print(f"Post Processing set to: {self.post_process}")
			if status:
				self.load_video()
			else:
				self.cleanup()

	def is_replay_active(self) -> bool:
		"""Thread-safe check of replay status."""
		with self.lock:
			return self.replay_active

	def frame_reader(self) -> None:
		"""Reads frames from the video and stores them in the buffer."""

		while not self.stop_buffer_thread:
			if self.get_frame_at_timestamp:
				time_in_buffer = self.cap.get(cv2.CAP_PROP_POS_MSEC)

				self.cap.set(cv2.CAP_PROP_POS_MSEC, self.cur_time_stamp - self.time_add_per_frame)
				_, frame = self.cap.read()

				self.cap.set(cv2.CAP_PROP_POS_MSEC, time_in_buffer)



				try:
					if self.post_process:
						frame = self.frame_processor.process_frame(frame)
				except TypeError:
					print("error processing... trying to continue")

				self.get_frame_at_timestamp = False
				self.frame_delivered = True
				self.save_frame = frame
			if not self.frame_buffer.full():
				ret, frame = self.cap.read()
				try:
					if self.post_process:
						frame = self.frame_processor.process_frame(frame)
				except TypeError:
					print("error processing... trying to continue")

				if not ret:
					self.stop_buffer_thread = True  # End of video
					break
				self.frame_buffer.put(frame)
			else:
				threading.Event().wait(0.01)


	def play(self) -> None:
		"""Main playback loop."""
		print("Base Controls:")
		print("Space: Pause/Play")
		print("Scroll up/ down: Zoom in/ out based on mouse position")
		print("W: Speed up")
		print("S: Slow down")
		print("Q: Quit")
		print("X: start post-processing")
		print("----- POST PROCESSING CONTROLS -----")
		print(self.frame_processor.get_control_text())

		while True:
			if not self.is_replay_active():
				# Show a blank frame when replay is inactive
				blank_frame = self.create_blank_frame()

				cv2.imshow('Video Player', blank_frame)
				key = cv2.waitKey(100) & 0xFF
				if key == ord('q'):
					self.cleanup()
					break
			# elif key == ord('/'):
			# 	self.receiving_person_left = not self.receiving_person_left
			# 	print(f"Receiver Left: {self.receiving_person_left}")
			#
			# continue

			# Playback active
			if not self.paused and not self.frame_buffer.empty():
				frame = self.frame_buffer.get()
				self.cur_time_stamp += self.time_add_per_frame
				frame = self.apply_zoom(frame)
				cv2.imshow('Video Player', frame)
				# print(self.frame_buffer.qsize())

			elif self.paused:  # allows zooming and moving whilst paused

				if self.frame_delivered:
					frame = self.save_frame
					frame = self.apply_zoom(frame)
				cv2.imshow('Video Player', frame)


			key = cv2.waitKey(self.playback_speed) & 0xFF
			if key == ord('q'):  # Quit
				self.cleanup()
				break
			elif key == 32:  # Space: Pause/Play
				self.paused = not self.paused
				if self.paused:
					self.frame_delivered = False
					self.get_frame_at_timestamp = True

			elif key == 82 or key == ord('w'):  # Up Arrow: Speed up
				self.playback_speed = max(1, self.playback_speed - 5)
			elif key == 84 or key == ord('s'):  # Down Arrow: Slow down
				self.playback_speed += 5
			elif key == ord('x'):
				self.post_process = not self.post_process
				self.frame_buffer = Queue(maxsize=self.buffer_size)
				print(f"Analysing High: {self.post_process}")
			else:
				self.frame_processor.key_action(key)

	def apply_zoom(self, frame: np.ndarray, zoom_factor_override: bool = False) -> np.ndarray:
		"""Apply zoom to the given frame."""
		if self.zoom_factor == 1.0 or zoom_factor_override:
			return frame

		h, w, _ = frame.shape
		center_x, center_y = self.zoom_center

		zoom_w, zoom_h = int(w / self.zoom_factor), int(h / self.zoom_factor)

		x1 = max(center_x - zoom_w // 2, 0)
		y1 = max(center_y - zoom_h // 2, 0)
		x2 = min(center_x + zoom_w // 2, w)
		y2 = min(center_y + zoom_h // 2, h)

		cropped_frame = frame[y1:y2, x1:x2]
		return cv2.resize(cropped_frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

	def create_blank_frame(self) -> np.ndarray:
		"""Create a blank frame to display when no video is loaded."""
		blank_frame = 255 * np.ones((HEIGHT, WIDTH, 3), dtype="uint8")  # HD resolution
		cv2.putText(blank_frame, "Waiting for Replay...", (100, 240),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
		return blank_frame

	def cleanup(self) -> None:
		"""Releases resources"""
		self.stop_buffer_thread = True
		if self.buffer_thread is not None:
			self.buffer_thread.join()

		if self.cap:
			self.cap.release()

		self.frame_buffer = Queue(maxsize=self.buffer_size)

		self.frame_processor.cleanup()
