from typing import Dict, Tuple

from ProcessFrame import ProcessFrame
import mediapipe as mp
import numpy as np
import cv2

class ShoulderDetect(ProcessFrame):
	def __init__(self, control_dict: Dict[str, str]):
		ProcessFrame.__init__(self, control_dict)

		# MediaPipe for Pose Detection and Annotation
		self.mp_pose = mp.solutions.pose
		self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=True,
									  smooth_segmentation=False)
		self.mp_drawing = mp.solutions.drawing_utils

	def process_frame(self, frame, **kwargs):
		left, y_prev, serve = kwargs.get('left', True), kwargs.get('y_prev', 1000000), kwargs.get('serve', False)

		# Convert the frame to RGB as MediaPipe expects RGB images
		image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		return frame

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
	def weighted_average_color_change(image: ndarray, x: int, y: int, n: int = 10,
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
