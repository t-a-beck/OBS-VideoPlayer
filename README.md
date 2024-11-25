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
Controls:
Space: Pause/Play
Arrow Up: Speed up
Arrow Down: Slow down
Q: Quit
Mouse Scroll: Zoom in/ out 
.: start high detection measuring point
/ Change receiver half detect
Z: Start zoom box
X: Start Player Analyse
```

## Technical Details: Post Processing Function
Here, we outline the structure and required methods for implementing a post-processing function  in the system denoted by the class `ProcessFrame`


### Construction
Per default, the constructor `ProcessFrame.ProcessFrame()` is called. If addtional arguments are to be passed, these can be done over the keyword arguments in the `VideoPlayer` Class Constructor.


### Required Methods
The **Post Processing Class** must implement the following methods to control and process frames within the video player system. These methods enable the manipulation of frames and potentially provides the user control over various relevant parameters.
#### 1. `get_control_keys()`

This method should return an array containing all relevant keys used for controlling the post-processing behavior.

- **Important**: Do **not** include any keys that are used by the `VideoPlayer` in this dictionary.
- The `VideoPlayer` class has a `processing_func_data` attribute, which can store and access relevant data. 

**Example**:
```python
class ProcessFrame:
    def __init__(self):
        ...
    
    def get_control_keys(self):
        return ["z","u","i"]
```

#### 2. `get_control_text()`

This method should return a text output describing the control options available for post-processing.

 - The text should summarize the relevant controls that the user can adjust, making it clear what actions are possible.

**Example**:
```python
class ProcessFrame:
    def __init__(self):
        ...
    
    def get_control_text(self):
        return """
        i: Increase brightness of the frame
        u: Decrease brightness of the frame
        z: Addtional functions  
        """
        
```

#### 3. `process_frame(frame, self.processing_func_data)`
This method should take an input frame and apply the desired post-processing manipulations. It returns a frame of the same size as the input frame, with the necessary modifications applied.

- The method should handle any specific processing logic as defined by the post-processing function instance.
- The input frame should remain consistent in size and format after processing.

**Example**:
```python
class ProcessFrame:
    def __init__(self):
        ...
    
    def process_frame(self, frame, **kwargs):
        brightness = kwargs.get('brightness', 50)
        contrast = kwargs.get('contrast', 50)
        # Apply brightness and contrast adjustments
        frame = self.adjust_brightness_contrast(frame, brightness, contrast)
        return frame
    
    def adjust_brightness_contrast(frame, brightness, contrast):
        ...
```


#### Notes

- **Processing Overhead**: While `processing_func_data` can be used to store intermediate data, try to minimize its usage to reduce the impact on performance.
- **Consistent Frame Size**: Ensure that the frame returned by `process_frame()` is the same size as the input frame.
- **Control Interface**: Provide an intuitive control interface to allow users to easily modify processing settings such as brightness, contrast, etc.
