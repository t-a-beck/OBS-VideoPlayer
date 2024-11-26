import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=True, smooth_segmentation=True)
mp_drawing = mp.solutions.drawing_utils


def lin_line(x1, x2, y1, y2):
	# Calculate slope (a) and intercept (b)
	if x2 == x1:
		a = 0
	else:
		a = (y2 - y1) / (x2 - x1)
	b = y1 - a * x1
	return a, b


def weighted_average_color_change(image, x, y, n=10, threshold=10):
	"""
    Compute a weighted average of n pixels above the given (x, y) point, giving more weight
    to pixels that are further away from (x, y).
    
    Parameters:
        image (numpy.ndarray): The input OpenCV image.
        x (int): The x-coordinate of the reference point.
        y (int): The y-coordinate of the reference point.
        n (int): The number of pixels to sample above (x, y).
        
    Returns:
        numpy.ndarray: The weighted average color as an array [B, G, R].
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


def process(frame, left, y_prev, serve):
	# Convert the frame to RGB as MediaPipe expects RGB images
	image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	cv2.rectangle(frame, (0, 0), (int(2 * frame.shape[1] / 3), int(frame.shape[0])), (0, 255, 0), 2)
	cv2.rectangle(frame, (int(1 * frame.shape[1] / 3), 0), (int(frame.shape[1]), int(frame.shape[0])), (0, 255, 255), 2)

	#     # Load the image using OpenCV
	# image_bgr = cv2.imread("IMG_D17AFFCC538D-1.jpeg")
	# frame = image_bgr
	# # Convert the image from BGR to RGB
	# image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

	# Process the frame for pose landmarks

	if left:
		image_side = image_rgb[:, :int(2 * frame.shape[1] / 3), :]
	else:
		image_side = image_rgb[:, int(1 * frame.shape[1] / 3):, :]

	result = pose.process(image_side)

	# Check if shoulders are detected
	if result.pose_landmarks and mp_pose.PoseLandmark.LEFT_SHOULDER and mp_pose.PoseLandmark.RIGHT_SHOULDER:

		for landmark in result.pose_landmarks.landmark:
			if landmark:
				# Shift x-coordinates based on the side
				landmark.x *= 2 / 3  # Shift the relative x-coordinate for the scaling as processing only gets fraction of original frame
				if not left:
					landmark.x += 1 / 3 # add base shift for right side
		mp_drawing.draw_landmarks(
			frame,
			result.pose_landmarks,
			mp_pose.POSE_CONNECTIONS,
			mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
			mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
		)
		# Get the y-coordinates of the left and right shoulder landmarks
		left_shoulder_y, left_shoulder_x = int(
			result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0]), int(
			result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1])

		right_shoulder_y, right_shoulder_x = int(
			result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0]), int(
			result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1])

		a, b = lin_line(left_shoulder_x, right_shoulder_x, -left_shoulder_y, -right_shoulder_y)
		# Find highest shoulder
		shoulder_y = int(min(left_shoulder_y, right_shoulder_y))

		# cv2.circle(frame, (left_shoulder_x, left_shoulder_y), radius=20, color=(0, 255, 0), thickness=3)
		# cv2.circle(frame, (right_shoulder_x, right_shoulder_y), radius=20, color=(0, 255, 0), thickness=3)

		segmentation_mask = result.segmentation_mask

		extra = np.zeros((frame.shape[0], frame.shape[1] - segmentation_mask.shape[1]))

		if left:
			segmentation_mask = np.hstack([segmentation_mask, extra])
		else:
			segmentation_mask = np.hstack([extra, segmentation_mask])

		# Normalize mask for visualization (optional depending on requirements)
		# Scaling to 0-255 range for display with OpenCV
		segmentation_mask_normalized = cv2.normalize(segmentation_mask, None, 0, 255, cv2.NORM_MINMAX)
		segmentation_mask_normalized = segmentation_mask_normalized.astype(np.uint8)

		mask_colored = np.stack([np.zeros_like(segmentation_mask_normalized), segmentation_mask_normalized,
								 np.zeros_like(segmentation_mask_normalized)], axis=-1)

		# Apply the segmentation mask to the frame
		# Masking the green areas where the segmentation mask has values > a threshold (e.g., > 50)

		detect_threshold = 0.3
		# Define ROIs around shoulders
		roi_width = int(0.01 * frame.shape[1])  # 10 % of frame width
		roi_height = int(0.01 * frame.shape[0])  # 10 % of frame height

		person_mask_coords = np.argwhere(segmentation_mask > detect_threshold)
		# Find top edge points in each ROI
		roi_left = segmentation_mask[left_shoulder_y - roi_height: left_shoulder_y + roi_height,
				   left_shoulder_x - roi_width:left_shoulder_x + roi_width]
		roi_right = segmentation_mask[right_shoulder_y - roi_height: right_shoulder_y + roi_height,
					right_shoulder_x - roi_width:right_shoulder_x + roi_width]

		left_pos = person_mask_coords[(person_mask_coords[:, 0] >= left_shoulder_y - roi_height) & (
				person_mask_coords[:, 0] <= left_shoulder_y + roi_height) &
									  (person_mask_coords[:, 1] >= left_shoulder_x - roi_width) & (
											  person_mask_coords[:, 1] <= left_shoulder_x - roi_width)]

		# Assuming the threshold for significant y-jump to ignore head detection
		max_height_jump = 10  # This is a chosen value; adjust based on your data
		# left_top = np.min(left_pos[:,0]) if np.any(segmentation_mask > detect_threshold) else 0
		# right_top = np.min(np.where(roi_right, roi_right, frame.shape[0])) if np.any(segmentation_mask > detect_threshold) else 0

		# Sort by y-coordinates to check continuity from bottom to top of ROI
		detect_threshold = 0.9

		# Find indices that satisfy the line equation within the tolerance
		satisfying_points = []
		top_y = 0
		# print(a,b)
		if a < 0:  # left shoulder up
			top_y = left_shoulder_y
			for x in range(max(0, left_shoulder_x - roi_width), left_shoulder_x):
				# Check if the point (i, j) satisfies the line equation within the tolerance
				y = min(-(a * x + b), frame.shape[0])
				if y <= left_shoulder_y and y >= left_shoulder_y - roi_height and segmentation_mask[
					int(y), x] > detect_threshold:
					top_y = min(int(y), top_y)
		# print("append")
		else:  # right shoulder up
			top_y = right_shoulder_y
			for x in range(right_shoulder_x, min(frame.shape[1], right_shoulder_x + roi_width)):
				y = min(-(a * x + b), frame.shape[0])

				if y <= right_shoulder_y and y >= right_shoulder_y - roi_height and segmentation_mask[
					int(y), x] > detect_threshold:
					top_y = min(int(y), top_y)

		# print(satisfying_points)
		satisfying_points = np.array(satisfying_points)
		# left_top = np.min(satisfying_points[:,0])
		# print(left_shoulder_y)
		# print(left_top)
		alpha = 0.0  # Transparency factor for blending

		# Draw a horizontal line at shoulder height
		# cv2.line(frame, (0, top_y), (frame.shape[1], top_y), (0, 0, 255), 1)

		# cv2.circle(frame, (left_shoulder_x, left_top), radius=5, color=(0, 0, 255), thickness=3)
		# cv2.circle(frame, (right_shoulder_x, right_top), radius=5, color=(0, 0, 255), thickness=3)

		y2 = weighted_average_color_change(image_rgb, left_shoulder_x, left_shoulder_y, n=20, threshold=3) - 15
		# cv2.line(frame, (0, y2), (frame.shape[1], y2), (255, 0, 0), 1)
		y3 = weighted_average_color_change(image_rgb, right_shoulder_x, right_shoulder_y, n=20, threshold=3) - 15
		# cv2.line(frame, (0, shoulder_y), (frame.shape[1], shoulder_y), (255, 255, 0), 1)

		frame_with_mask = cv2.addWeighted(frame, 1 - alpha, mask_colored, alpha, 0)

		if a < 0.1:
			y_main = np.mean([shoulder_y - 30, y2, y3])
		else:
			y_main = np.mean([shoulder_y, top_y, y2, y3])
		y_main = int(shoulder_y - 15)

		if serve:
			y_main = min(y_prev, y_main)
		cv2.line(frame_with_mask, (0, y_main), (frame.shape[1], y_main), (255, 0, 255), 4)
		# Create a gradient to simulate depth (3D effect)

		# for i in range(10):
		#     # Calculate start and end points of the line for each depth level
		#     x_start, x_end = 0, frame.shape[1]
		#     x_start, x_end = left_shoulder_x - roi_width, left_shoulder_x
		#     y_start = int(a * x_start + b + i * 3)  # Offset by i to create 3D effect
		#     y_end = int(a * x_end + b + i * 3)

		#     # Define color and thickness to enhance the 3D effect
		#     color = (255 - i * 20, 100 + i * 15, 100 + i * 15)  # Varying colors for depth
		#     thickness = 1 + i  # Increase thickness with depth

		#     # Draw the line with offset
		#     cv2.line(frame_with_mask, (x_start, y_start), (x_end, y_end), color, thickness)

		x_start, x_end = left_shoulder_x - roi_width, left_shoulder_x
		x_start, x_end = 0, frame.shape[1]
		y_start = int(a * x_start + b)
		y_end = int(a * x_end + b)
		# cv2.line(frame_with_mask, (x_start, -y_start), (x_end, -y_end), (0, 0, 255), 2)

		# cv2.imshow("Original Image with Segmentation Mask", frame_with_mask)

		return frame_with_mask, y_main
