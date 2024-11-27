from typing import Dict, Set
from numpy import ndarray


class ProcessFrame:
	def __init__(self, control_dict: Dict[str, str] = {}):
		self.control_text_per_key: Dict[str, str] = control_dict  # Maps control key to relevant explanation text
		self.control_keys: Set[str] = set(self.control_text_per_key.keys())
		self.__control_text__: str = self.make_control_text()

	def make_control_text(self) -> str:
		out = ""
		for key in self.control_keys:
			out += f"{key}: {self.control_text_per_key.get(key, "")} \n"

		if out == "":
			out = "No controls"

		return out

	def get_control_text(self) -> str:
		return self.__control_text__

	def key_action(self, key) -> None:
		pass

	def process_frame(self, frame: ndarray) -> ndarray:
		return frame

	def cleanup(self) -> None:
		pass


if __name__ == "__main__":
	pass
