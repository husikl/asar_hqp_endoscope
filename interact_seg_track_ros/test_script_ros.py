import rospy
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import Header
import numpy as np
import cv2
import time
import threading
from queue import Queue
from std_srvs.srv import Empty
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseArray, Pose
import tkinter as tk
from main import build_control, init_interactive_segmentation, inference_masks

res_manager, interact_control = build_control()


class ResourceHandler:
    def __init__(
        self,
        command_queue,
        initial_image,
        res_manager,
        interact_control,
        tk_root,
        masks_queue,
    ):
        self.command_queue = command_queue
        self.initial_image = initial_image
        self.res_manager = res_manager
        self.interact_control = interact_control

        self.lock = threading.Lock()
        self.image_to_process = None
        self.masks_result = None
        self.tk_root = tk_root
        self.masks_queue = masks_queue
       
    def process_commands(self):
        while True:
            if not self.command_queue.empty():
                command, image = self.command_queue.get()
                if command == "init_segmentation":
                    self.init_segmentation(image)
                elif command == "infer_masks":
                    self.infer_masks(image)
            rospy.sleep(1.0/65.0)

    def init_segmentation(self, image):
        with self.lock:
            interact_control.image_queue.put(image)
            init_interactive_segmentation(
                image, res_manager, interact_control, self.tk_root
            )
            # self.masks_queue.put(self.masks_result)

    def infer_masks(self, image):
        # print("got start request")
        with self.lock:
            self.masks_result = inference_masks(image, res_manager)
            self.masks_queue.put(self.masks_result)
        # print("end ..")


class ImageProcessor:
    def __init__(self, command_queue, masks_queue):
        self.command_queue = command_queue
        self.bridge = CvBridge()
        self.initial_image_collected = threading.Event()
        self.image_sub = rospy.Subscriber(
            "/ximea_cam/image_raw", ImageMsg, self.image_callback
        )
        self.init_service = rospy.Service(
            "init_segmentation", Empty, self.init_segmentation_service
        )
        self.inference_service = rospy.Service(
            "run_inference", Empty, self.run_inference_service
        )
        self.detected_tools_pub = rospy.Publisher(
            "tracked_tools", PoseArray, queue_size=1
        )

        self.latest_image = None
        self.masks_queue = masks_queue
        self.processing_enabled = False
        self.init_segmentation_done = False
        self.lock = threading.Lock()  # Initialize the lock here
        self.processing_thread = threading.Thread(target=self.process_images)
        self.processing_thread.start()
        self.scale = 3.0

        rospy.loginfo("init completed...")

    def process_images(self):
        msg = PoseArray()
        header = Header()
        
        # n = 3  # Number of expected objects
        # empty_pose = Pose() 
        # msg.poses = [empty_pose] * n
        while True:
            with self.lock:
                if (self.processing_enabled and self.init_segmentation_done and self.latest_image is not None):
                    # t1 = time.time()
                    self.command_queue.put(("infer_masks", self.latest_image))

                    # Wait for the result to be put into the queue
                    masks_result = self.masks_queue.get()
                    
                    # Sort keys to maintain a consistent order
                    sorted_keys = sorted(masks_result.keys())
                    msg.poses.clear()
                    for idx, key in enumerate(sorted_keys):
                        value = masks_result[key]
                        p = Pose()
                        if value[2] != 0:  # Check if the object was detected
                            x, y = self.map_coordinates_to_original((value[0], value[1]))
                            p.position.x = x
                            p.position.y = y
                            p.position.z = value[2]*self.scale
                            p.orientation.x = idx+1
                        else:
                            p.orientation.x = -(idx+1)  
                            # print("not tracking : ", idx+1)
                        msg.poses.append(p)
                    msg.header.stamp = rospy.Time.from_sec(time.time())
                    self.detected_tools_pub.publish(msg)
                    # print(f"number of tools detected: {len(msg.poses)}")

                    self.latest_image = None  # Clear the latest image
                    # t2 = time.time()
                    # print("loop time", t2-t1)

            rospy.sleep(1.0 / 65.0)

    def map_coordinates_to_original(self, coordinates):
        """
        Maps coordinates from the resized ROI back to the original image.

        Args:
            coordinates: tuple (x, y) in resized ROI
            roi_offset: tuple (x_offset, y_offset) of the ROI in original image
            roi_dims: tuple (roi_width, roi_height) dimensions of the ROI
            new_dims: tuple (new_width, new_height) dimensions of the resized ROI
            original_dims: tuple (original_width, original_height) dimensions of the original image

        Returns:
            tuple (x, y) mapped to the original image
        """
        roi_offset = (392, 0)
        # roi_dims = (1264, 1088)  # width, height of ROI
        # new_dims = (roi_dims[0] // 2, roi_dims[1] // 2)
        # original_dims = (2048, 1088)  # original image dimensions
        x, y = coordinates
        x_offset, y_offset = roi_offset
        # roi_width, roi_height = roi_dims
        # new_width, new_height = new_dims
        # original_width, original_height = original_dims

        # x_scale = roi_width / new_width
        # y_scale = roi_height / new_height

        # x_original = int(x * 2) + x_offset
        # y_original = int(y * 2) + y_offsetv
        x_original = int(x * 4) + x_offset
        y_original = int(y * 4) + y_offset

        return x_original, y_original

    def compute_r(self, contour):
        # Compute the radius of the circle that encloses the contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        return radius

    def image_callback(self, msg):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # extract ROI
            extracted_image = self.current_image[0 : 0 + 1088, 392 : 392 + 1264]

            # Resize the extracted image to speed up detections
            new_dims = (extracted_image.shape[1] // 4, extracted_image.shape[0] // 4)
            resized_extracted_image = cv2.resize(extracted_image, new_dims)
            # print("shape : ", resized_extracted_image.shape)

            # Store the resized extracted image
            self.resized_image = resized_extracted_image

            self.latest_image = resized_extracted_image
            if not self.initial_image_collected.is_set():
                self.initial_image = resized_extracted_image
                self.initial_image_collected.set()
        except CvBridgeError as e:
            print(e)

    def init_segmentation_service(self, req):
        self.command_queue.put(("init_segmentation", self.latest_image))
        self.init_segmentation_done = True
        return []

    def run_inference_service(self, req):
        if not self.init_segmentation_done:
            print("Initialization not done. Ignoring start request.")
            return []

        self.processing_enabled = not self.processing_enabled  # Toggle processing
        return []


if __name__ == "__main__":
    rospy.init_node("image_processor_node", anonymous=False)

    command_queue = Queue()
    masks_queue = Queue()
    # Initialize ImageProcessor
    image_processor = ImageProcessor(command_queue, masks_queue)

    # Wait until an initial image is collected
    image_processor.initial_image_collected.wait()

    tk_root = tk.Tk()
    # Initialize ResourceHandler with res_manager and interact_control
    resource_handler = ResourceHandler(
        command_queue,
        image_processor.initial_image,
        res_manager,
        interact_control,
        tk_root,
        masks_queue,
    )

    resource_thread = threading.Thread(target=resource_handler.process_commands)
    resource_thread.start()
    tk_root.mainloop()

    rospy.spin()
