from typing import Tuple

from ProcessFrame import ProcessFrame
import mediapipe as mp
import numpy as np
import cv2


class ShoulderDetect(ProcessFrame):
	def __init__(self):
		control_dict = {
			".": "start high detection measuring point",
			"/": "Change receiver half detect",
			"X": "Highlight receiver with pose detection"
		}

		ProcessFrame.__init__(self, control_dict)

		# MediaPipe for Pose Detection and Annotation
		self.mp_pose = mp.solutions.pose
		self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=True,
									  smooth_segmentation=True)
		self.mp_drawing = mp.solutions.drawing_utils

		# Shifting for frames if only portions of frame are fed into pose detection
		self.x_offset = 1 / 3
		self.x_factor = 2 / 3
		self.y_offset = 0
		self.y_factor = 1

		# Other params
		self.serve = False
		self.left = False
		self.y_prev = 1000000
		self.pose_highlight = True


	def key_action(self, key) -> None:
		if key in self.control_keys:
			if key == ord("."):
				self.serve = not self.serve
				print(f"Serve started: {self.serve}")
			elif key == ord("/"):
				self.left = not self.left
				print(f"Receiver Left: {self.left}")
			elif key == ord("X"):
				self.pose_highlight = not self.pose_highlight
				print(f"Receiver Pose Highlight: {self.pose_highlight}")

	def process_frame(self, frame):
		# Convert frame to RGB for MediaPipe
		image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		image_side = self.extract_image_section(frame, image_rgb, self.left)

		pose_result = self.pose.process(image_side)

		# Check if shoulders are detected
		if pose_result.pose_landmarks:
			self.reshift_landmarks_to_original(self.left, pose_result)

			if self.pose_highlight:
				self.draw_pose(frame, pose_result)

			# Get the coordinates of the left and right shoulder landmarks
			left_shoulder_y, left_shoulder_x, right_shoulder_y, right_shoulder_x = self.extract_shoulders(frame,
																										  pose_result)


			cur_shoulder = self.find_shoulder_line(frame, left_shoulder_y, left_shoulder_x, right_shoulder_y,
											 right_shoulder_x)

			if self.serve:
				y_actual = min(self.y_prev, cur_shoulder)

			self.y_prev = y_actual

			# Draw shoulder line
			cv2.line(frame, (0, y_actual), (frame.shape[1], y_actual), (255, 0, 255), 4)

		return frame

	def find_shoulder_line(self, frame, left_shoulder_y, left_shoulder_x, right_shoulder_y,
						   right_shoulder_x):

		shoulder_offset = self.find_higher_shoulder_with_shift(left_shoulder_y, right_shoulder_y)



		return shoulder_offset

	def find_higher_shoulder_with_shift(self, left_shoulder_y, right_shoulder_y, shift=15):

		shoulder_y = int(min(left_shoulder_y, right_shoulder_y)) - shift

		return shoulder_y

	def extract_shoulders(self, frame, pose_result):
		try:
			left_shoulder_y, left_shoulder_x = int(
				pose_result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0]), int(
				pose_result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1])
			right_shoulder_y, right_shoulder_x = int(
				pose_result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0]), int(
				pose_result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1])
		except AttributeError:
			print("some shoulders could not be found, attempting to continue")

		return left_shoulder_y, left_shoulder_x, right_shoulder_y, right_shoulder_x

	def draw_pose(self, frame, result):
		self.mp_drawing.draw_landmarks(
			frame,
			result.pose_landmarks,
			self.mp_pose.POSE_CONNECTIONS,
			self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
			self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
		)

	def reshift_landmarks_to_original(self, result, left, top=True):
		for landmark in result.pose_landmarks.landmark:
			if landmark:
				# Shift coordinates back
				landmark.x *= self.x_factor  # Shift the relative x-coordinate for the scaling as processing only gets fraction of original frame
				landmark.y *= self.y_factor
				if not left:
					landmark.x += self.x_offset  # add base shift for right side
				if not top:
					landmark.y += self.y_offset

	def extract_image_section(self, frame, image_rgb, left, top=True):
		# Segment the image into the respective 2/3rds sections to make person localisation easier
		x_start = 0
		x_end = frame.shape[1]

		y_start = 0
		y_end = frame.shape[0]

		if left:
			x_end = int(self.x_factor * frame.shape[1])
		else:
			x_start = int(self.x_offset * frame.shape[1])

		if top:
			y_end = int(self.y_factor * frame.shape[0])
		else:
			y_start = int(self.y_offset * frame.shape[0])

		return image_rgb[y_start:y_end, x_start:x_end, :]

	@staticmethod
	def lin_line(x1: float, x2: float, y1: float, y2: float) -> Tuple[float, float]:
		"""
		Calculate slope (a) and intercept (b) with linear line between 2 points. If x2 == x1, the 0 line is returned i.e. 0*x + 0

		Parameters:
			x1: float
			x2: float
			y1: float
			y2: float

		Returns:
			Slope a and intercept b
		"""
		if x2 == x1:
			a = 0
			b = 0
		else:
			a = (y2 - y1) / (x2 - x1)
			b = y1 - a * x1
		return a, b

	@staticmethod
	def weighted_average_color_change(image: np.ndarray, x: int, y: int, n: int = 10,
									  threshold: float = 10.0) -> int:
		"""
		Compute a weighted average of n pixels above the given (x, y) point, giving more weight
		to pixels that are further away from (x, y).

		Parameters:

			image (numpy.ndarray): The input OpenCV image.
			x (int): The x-coordinate of the reference point.
			y (int): The y-coordinate of the reference point.
			n (int): The number of pixels to sample above (x, y).
			threshold (float): Threshold for colour change

		Returns:
			int: The y coordinate of the reference point where a significant colour change  (as determined by the threshold) is met
		"""
		# Initialize weighted color sum and total weight
		weighted_sum = 0.0
		total_weight = 0.0

		reference_color = image[y, x].astype(float)
		# Traverse n pixels above (x, y) or until we reach the top of the image
		found = False
		while not found:
			y -= 1
			if y < 0:
				break

			for i in range(1, n + 1):
				# Calculate the current y-coordinate
				current_y = y - i
				if current_y < 0:
					break  # Stop if we go beyond the top edge

				# Fetch the color at the current (x, current_y) position
				current_color = image[current_y, x].astype(float)
				color_diff = np.linalg.norm(reference_color - current_color)
				# Define weight as the distance from the starting point (higher weight for pixels further up)
				weight = i
				weighted_sum += color_diff * weight
				total_weight += weight

			# Calculate weighted average color
			if total_weight == 0:
				return image[y, x]  # Return the original color if no pixels are sampled
			weighted_average = weighted_sum / total_weight

			if weighted_average > threshold:
				break

		return y
