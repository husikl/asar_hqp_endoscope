import rospy
from sensor_msgs.msg import Image as ImageMsg
import numpy as np
from PIL import Image
import cv2
from main import build_control, init_interactive_segmentation, inference_masks
import time


# Initialize video reader and writer
input_video_path = "test_video2.mp4"
output_video_path = "output_video2.mp4"


cap = cv2.VideoCapture(input_video_path)

# Seek to the 1 minute 33-second mark
cap.set(cv2.CAP_PROP_POS_MSEC, 30000)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    output_video_path, fourcc, 20.0, (2048, 1088)
)  # Adjust frame size

# Initialize segmentation
res_manager, interact_control = build_control()
ret, frame = cap.read()
if ret:
    masks, circle = init_interactive_segmentation(frame, res_manager, interact_control)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t0 = time.time()
    masks, circle = inference_masks(frame, res_manager)
    t1 = time.time()

    print(t1 - t0, "second")

    # Apply masks to the frame
    for mask in masks[1:]:  # Assuming masks[0] is not needed
        frame += cv2.cvtColor(
            (mask.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
        )

    # Write frame with masks to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print("finish")
