import unittest

import numpy as np

from ProcessFrame import ProcessFrame
class ProcessFrameTest(unittest.TestCase):
	def test_construct_standard(self):
		dict_standard = {
			".": "start high detection measuring point",
			"/": "Change receiver half detect",
			"X": "Start Player Analyse"
		}
		processor = ProcessFrame(dict_standard)

		self.assertEqual(set(processor.get_control_keys()), set([".","/","X"]))  # add assertion here
	def test_construct_empty(self):
		dict_empty = {}
		processor = ProcessFrame(dict_empty)

		self.assertEqual(processor.get_control_keys(), [])  # add assertion here
		self.assertEqual(processor.get_control_text(), "")  # add assertion here

	def test_frame_size(self):
		dict_empty = {}
		frame = 255 * np.ones((1080, 1920, 3), dtype="uint8")
		processor = ProcessFrame(dict_empty)
		self.assertEqual(processor.process_frame(frame).shape, frame.shape)  # add assertion here


if __name__ == '__main__':
	unittest.main()
