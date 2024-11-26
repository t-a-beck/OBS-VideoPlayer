from typing import List, Dict
import cv2
from numpy import ndarray
import numpy as np

class ProcessFrame:
	def __init__(self, control_dict: Dict[str, str]):

		self.control_text_per_key: Dict[str, str] = control_dict  # Maps control key to relevant explanation text
		self.control_keys: List[str] = list(self.control_text_per_key.keys())
		self.control_text: str = self.make_control_text()

	def make_control_text(self) -> str:
		out = ""
		for key in self.control_keys:
			out += f"{key}: {self.control_text_per_key.get(key, "")} \n"
		return out

	def get_control_keys(self) -> List[str]:
		return self.control_keys

	def get_control_text(self) -> List[str]:
		return self.control_text

	def process_frame(self, frame: ndarray, **kwargs) -> ndarray:
		colour = kwargs.get("colour", 0)
		new_colour = np.random() + colour

		# Convert from BGR to RGB as this is more conventional
		image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		return image_rgb, {"colour": new_colour}

if __name__ == "__main__":
	pass