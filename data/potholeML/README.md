# Roadway Quality ML #

## Data Set Generation ##

### Video Preprocessing Script ###
FILE DEPENDENCIES
    * OpenCV    (`import cv2`)
        * `pip install opencv-python` if cv2 not found
    * os        (`import os`)

To add more images to the folders, just add corresponding video files to `Videos` directory and the change directory to parent directory and run `unpackVideoScript.py`.

The file `unpackVideoScript.py` unpacks individual frames from videos in directories within `Videos` directory. Based on folder the video is placed in, the frames of that video are automatically processed and places within the corresponding `Images` subdirectory. If image files with corresponding video file names already exist in the `Images` subdirectory (ex: `test.mp4` and `test_0.png`), the new frames are not saved and the video file is ignored. To change video and image file extensions, change constants `VIDEO_EXT` and `IMG_EXT` to necessary extensions (ex: `".txt"`).
