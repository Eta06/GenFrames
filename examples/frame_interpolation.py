"""
Example: Interpolate individual frames
"""

import cv2
import numpy as np
from genframes import FrameInterpolator

# Initialize interpolator
interpolator = FrameInterpolator(model="rife", backend="auto")

# Load two frames
frame1 = cv2.imread("frame1.jpg")
frame2 = cv2.imread("frame2.jpg")

# Convert BGR to RGB
frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

# Interpolate 3 intermediate frames
# Returns: [frame1, intermediate1, intermediate2, intermediate3, frame2]
frames = interpolator.interpolate_frames(frame1, frame2, num_intermediate=3)

# Save interpolated frames
for i, frame in enumerate(frames):
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"output_frame_{i}.jpg", frame_bgr)

print(f"Generated {len(frames)} frames")
