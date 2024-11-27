# OBS Integrated Video Player
Implements a naive and simple Video Player that has suitable OBS integration and many useful features that make it useful for Streaming. It was developed for the [Roundnet Germany Stream](https://www.twitch.tv/roundnetgermany). 

Useful features include:
 - Ability to slow down, speed up and pause video play.
 - Post processing of invididual frames (for example for high detection for Roundnet see [link to other project]())
 - Buffering of frames for smoother play
 - Zoom in and out feature on the frames to highlight certain plays
 - Suitable OBS integration using [obs-websocket-py](https://github.com/Elektordi/obs-websocket-py)  to allow the VideoPlayer to play content based off the current scene
### Dependencies
The primary dependencies are  [obs-websocket-py](https://github.com/Elektordi/obs-websocket-py) and the relevant [OpenCV](https://opencv.org) Python bindings and it runs on `Python 3.12`.
## Installation
### Environment
It is recommended to start by installing [miniconda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) if not already done so, unless an alternative package manager wants to be used.

After installation, run 
```commandline
conda env create -f environment.yml
```

### OBS-Studio Configuration
#### WebSocket Server

1. Open OBS and navigate to **Tools** → **WebSocket Server Settings**.
2. Under the heading **Plugin Settings**, check that **Enable WebSocket server** is checked.
3. Set the **Server Port** to `4455`.
4. Disable **Authentication** (since it is unnecessary on a local server).
   

   If you'd like to use authentication, you can copy the relevant password into the `OBS_PASSWORD` parameter in the linked file [`obs-call.py`](obs-call.py).

#### Scenes
Change the `TARGET_SCENE` and `POST_PROCESS_REPLAY_SCENE` parameter in the linked file [`obs-call.py`](obs-call.py) to their respective counterparts OBS Scene name.

Now run the relevant execution command like in the usage section (for Unix systems replace & with &&)
```commandline
conda activate OBS-VideoPlayer & python obs-call.py
```

Ignore the window that opens and add a new **Window Capture Source** in OBS. It should be called `[python] Video Player`.
## Usage
First, ensure all steps from the instillation have been completed.

Next, activate the relevant conda environment with 
```commandline
conda activate OBS-VideoPlayer
```
Next, ensure that OBS is running and run (in the same terminal from before)
```commandline
python obs-call.py
```
The relevant VideoPlayer window should now be visible.


The principal controls are listed in the terminal when starting the program. They are also listed here: Note that these **only** work when the relevant Video is in focus (i.e. is selected).
```
Base Controls:
Space: Pause/Play
Scroll up/ down: Zoom in/ out based on mouse position
W: Speed up
S: Slow down
Q: Quit
X: start post-processing
----- POST PROCESSING CONTROLS -----
...
```

## Technical Details: 

### Post Processing Function
Here, we outline the structure and required methods for implementing a post-processing function in the video player system. Fundamentally, all custom post-processing functions should be a subtype of the class `ProcessFrame` and overwrite `ProcessFrame.process_frame(frame)`. 

This object is passed to the VideoPlayer during construction. Note that, per default, the frame processor is reconstructed to reset it when loading a video. This can be avoided by setting `reset_processor_on_load = False` in the `VideoPlayer` constructor  


#### Construction
Per default, the constructor `ProcessFrame.ProcessFrame()` is called by providing the relevant control dictionary (i.e. `string_key -> text_to_explain_key_function`).

**Important Notes**: 
- If the default constructor is overwritten, the super constructor should still be called to initialise control text correctly. these can be done over the keyword arguments in the `VideoPlayer` Class Constructor. An example is given in `ShoulderDetect.py`
- Do **not** include any keys that are used by the `VideoPlayer` in the control dictionary, otherwise these inputs will never be passed on correctly.
- The input frame **must** remain consistent in size and format after processing.

#### Example

```python
from frame_processing.ProcessFrame import ProcessFrame
import numpy as np


class CustomFrameProcessor(ProcessFrame):
    def __init__(self):
        control_dict = {".": "Increase max random number"}
        self.max_rand = 10
        ...
        
        # Make sure to call super constructor!
        ProcessFrame.__init__(control_dict)
        
    def process_frame(self, frame):
        brightness = np.random.randint(0, self.max_rand)
        new_frame = brightness * np.ones_like(frame)
        return new_frame

```
