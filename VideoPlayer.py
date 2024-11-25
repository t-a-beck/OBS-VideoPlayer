import cv2
import threading
import queue
import numpy as np
import platform
DEFAULT_VID_PATH = "test/Open_af_Alter-Rally!.mp4"  # Set to 0 for default webcam
HEIGHT = 1080
WIDTH = 1920
from shoulderFind import process


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

				self.zoom_center = (x,y)

	def __init__(self, buffer_size=100):
		# Initialize playback window and basic variables
		cv2.namedWindow('Video Player')
		self.cap = None
		self.video_path = None
		self.paused = True
		self.playback_speed = 30  # in milliseconds

		# Buffering of frames
		self.buffer_size = buffer_size
		self.frame_buffer = queue.Queue(maxsize=self.buffer_size)
		self.stop_buffer_thread = False
		self.buffer_thread = None
		self.lock = threading.Lock()

		# Replay status
		self.replay_active = False
		self.receiving_person_left = True
		self.save_highest = False
		self.highest_y = 100000000
		self.post_process = False
		self.save_frame = None


		#Zooming
		self.zoom_factor = 1.0
		self.zoom_center = (WIDTH // 2, HEIGHT // 2)  # Absolute center (x, y)
		cv2.setMouseCallback('Video Player', self.mouse_callback)

	def load_video(self, video_path=DEFAULT_VID_PATH):
		"""Load the video file and initialize playback."""
		self.video_path = video_path
		self.cap = cv2.VideoCapture(video_path)
		if not self.cap.isOpened():
			raise ValueError("Error: Cannot open the video file.")
		self.paused = False

		# Start buffering
		self.stop_buffer_thread = False
		self.buffer_thread = threading.Thread(target=self.frame_reader)
		self.buffer_thread.daemon = True
		self.buffer_thread.start()

		self.save_highest = False
		self.highest_y = 100000000

	def set_replay_status(self, status=False, post_process=False):
		"""External programs call this to update replay status."""
		with self.lock:
			self.replay_active = status
			print(f"Replay status set to: {self.replay_active}")
			self.post_process = post_process
			print(f"Analyse High set to: {self.post_process}")
			if status:
				self.load_video()
			else:
				self.cleanup()

	def is_replay_active(self):
		"""Thread-safe check of replay status."""
		with self.lock:
			return self.replay_active

	def frame_reader(self):
		"""Reads frames from the video and stores them in the buffer."""
		while not self.stop_buffer_thread:
			if not self.frame_buffer.full():
				ret, frame = self.cap.read()
				try:
					if self.post_process:
						frame, new_y = process(frame, self.receiving_person_left, self.highest_y, self.save_highest)
						self.highest_y = new_y
				except TypeError:
					print("error processing... trying to continue")

				if not ret:
					self.stop_buffer_thread = True  # End of video
					break
				self.frame_buffer.put(frame)
			else:
				threading.Event().wait(0.01)

	def play(self):
		"""Main playback loop."""
		print("Controls:")
		print("Space: Pause/Play")
		print("Arrow Up: Speed up")
		print("Arrow Down: Slow down")
		print("Q: Quit")
		print(". start high detection measuring point")
		print("/ Change receiver half detect")
		print("Z: Start zoom box")
		print("X: Start Player Analyse")

		while True:
			if not self.is_replay_active():
				# Show a blank frame when replay is inactive
				blank_frame = self.create_blank_frame()
				
				cv2.imshow('Video Player', blank_frame)
				key = cv2.waitKey(100) & 0xFF
				if key == ord('q'):
					self.cleanup()
					break
				elif key == ord('/'):
					self.receiving_person_left = not self.receiving_person_left
					print(f"Receiver Left: {self.receiving_person_left}")

				continue

			# Playback active
			if not self.paused and not self.frame_buffer.empty():
				frame = self.frame_buffer.get()
				frame = self.apply_zoom(frame)
				cv2.imshow('Video Player', frame)

			elif self.paused: # allows zooming and moving whilst paused 
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
					self.save_frame = frame
			elif key == 82 or key == ord('w'):  # Up Arrow: Speed up
				self.playback_speed = max(1, self.playback_speed - 5)
			elif key == 84 or key == ord('s'):  # Down Arrow: Slow down
				self.playback_speed += 5
			elif key == ord('.'):
				self.save_highest = not self.save_highest
				self.highest_y = 1000000
				if self.save_highest:
					print("RECORDING HIGHESTPOINT")
			elif key == ord('/'):
				self.receiving_person_left = not self.receiving_person_left
				print(f"Receiver Left: {self.receiving_person_left}")
			elif key == ord('X'):
				self.post_process = not self.post_process
				self.frame_buffer = queue.Queue(maxsize=self.buffer_size)
				print(f"Analysing High: {self.post_process}")


		

	def apply_zoom(self, frame):
		"""Apply zoom to the given frame."""
		if self.zoom_factor == 1.0:
			return frame

		h, w, _ = frame.shape
		center_x, center_y = self.zoom_center

		zoom_w, zoom_h = int(w / self.zoom_factor), int(h / self.zoom_factor)

		x1 = max(center_x - zoom_w // 2, 0)
		y1 = max(center_y - zoom_h // 2, 0)
		x2 = min(center_x + zoom_w // 2, w)
		y2 = min(center_y + zoom_h // 2, h)

		cropped_frame = frame[y1:y2, x1:x2]
		# print(cropped_frame)
		# print(cropped_frame.shape)
		return cv2.resize(cropped_frame, (WIDTH,HEIGHT), interpolation=cv2.INTER_LINEAR)




	def create_blank_frame(self):
		"""Create a blank frame to display when no video is loaded."""
		blank_frame = 255 * np.ones((HEIGHT, WIDTH, 3), dtype="uint8")  # HD resolution
		cv2.putText(blank_frame, "Waiting for Replay...", (100, 240),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
		return blank_frame

	def cleanup(self):
		"""Releases resources"""
		self.stop_buffer_thread = True
		if self.buffer_thread is not None:
			self.buffer_thread.join()

		if self.cap:
			self.cap.release()

		self.frame_buffer = queue.Queue(maxsize=self.buffer_size)
