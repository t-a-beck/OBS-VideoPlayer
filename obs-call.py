from obswebsocket import obsws, events
from VideoPlayer import VideoPlayer

# Modify as relevant:
OBS_HOST = "localhost"
OBS_PORT = 4455  # Default WebSocket port
OBS_PASSWORD = "very_secret"  # Replace with OBS WebSocket password (if wanted)

TARGET_SCENE_1 = "Replay 1"
TARGET_SCENE_2 = "Replay 2"
TARGET_SCENE_3 = "Replay 3"

# Don't change below this point unless you know what you're doing


def on_switch_scene(message):
	"""
	Callback for OBS scene switch events.
	"""
	print(f"Received Scene change: {message.getSceneName()}")
	if message.getSceneName() == TARGET_SCENE_1:
		player.set_replay_status(True, False, cam_number=1)
	elif message.getSceneName() == TARGET_SCENE_2:
		player.set_replay_status(True, False, cam_number=2)
	elif message.getSceneName() == TARGET_SCENE_3:
		player.set_replay_status(True, False, cam_number=3)
	else:
		player.set_replay_status(False, False)  # Deactivate replay


def main():
	"""
	Main function to connect to OBS WebSocket and listen for events.
	"""

	ws = obsws(OBS_HOST, OBS_PORT, OBS_PASSWORD)
	ws.register(on_switch_scene, events.CurrentProgramSceneChanged)
	ws.connect()

	# Construct VideoPlayer Object with shoulder detector as frame processing function

	global player
	player = VideoPlayer()
	player.play()

	try:
		print(f"Listening for scene changes... (Target: '{TARGET_SCENE_1}', '{TARGET_SCENE_2}', '{TARGET_SCENE_3}')")
		# Keep the script running to listen for events
		input("Press Enter to stop the script.\n")
	except KeyboardInterrupt:
		pass
	finally:
		ws.disconnect()
		player.cleanup()


if __name__ == "__main__":
	main()
