from obswebsocket import obsws, events
from VideoPlayer import VideoPlayer
from frame_processing.ShoulderDetect import ShoulderDetect
# Configuration
OBS_HOST = "localhost"
OBS_PORT = 4455  # Default WebSocket port
OBS_PASSWORD = "very_secret"  # Replace with OBS WebSocket password if wanted

TARGET_SCENE = "Replay"
POST_PROCESS_REPLAY_SCENE = "Replay"  # _POST_PROCESS


def on_switch_scene(message):
	"""
	Callback for OBS scene switch events.
	"""
	print(f"Received Scene change: {message.getSceneName()}")
	if message.getSceneName() == TARGET_SCENE:
		player.set_replay_status(True, False)
	elif message.getSceneName() == POST_PROCESS_REPLAY_SCENE:
		player.set_replay_status(True, True)  # also add post_processing
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
	player = VideoPlayer(frame_processor=ShoulderDetect())
	player.play()

	try:
		print(f"Listening for scene changes... (Target: '{TARGET_SCENE}')")
		# Keep the script running to listen for events
		input("Press Enter to stop the script.\n")
	except KeyboardInterrupt:
		pass
	finally:
		ws.disconnect()
		player.cleanup()


if __name__ == "__main__":
	main()
