import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from scipy.optimize import minimize
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Pose, Twist, Point, PoseArray
from std_srvs.srv import Empty, EmptyResponse
import random
from tf.transformations import (
    euler_from_quaternion,
    quaternion_from_euler,
    euler_from_matrix,
    euler_matrix,
)
from copy import deepcopy
import random
import numpy as np
import copy
import math
from scipy.stats import multivariate_normal
from cv2 import line, ellipse
from scipy.stats import entropy
from scipy.special import softmax
import time
from scipy.cluster.hierarchy import linkage, fcluster
from itertools import combinations


class SurgTip:
    def __init__(self, x: float, y: float, r: float, gamma: float = 1.0):
        self.x = x
        self.y = y
        self.r = r
        self.gamma = gamma


class ActiveInferenceVisualServo:
    def __init__(self):
        self.object_green = None
        self.object_red = None
        self.image_subscriber = rospy.Subscriber(
            "/ximea_cam/image_raw", Image, self.image_callback
        )
        self.tools_pub_ = rospy.Publisher("/tracked_tools", PoseArray, queue_size=1)
        # self.image_subscriber = rospy.Subscriber('/image', Image, self.image_callback)
        self.pub_green = rospy.Publisher("/green_object", Point, queue_size=1)

    def service_callback(self, request):
        # This is the ROS service callback that gets triggered when the service is called.
        # It will run the active inference control method and then return an empty response.
        print("request received")
        self.active_inference_control()
        return EmptyResponse()

    def image_callback(self, image_msg):
        self.current_image = CvBridge().imgmsg_to_cv2(
            image_msg, desired_encoding="bgr8"
        )
        img = self.detect_green_objects(self.current_image)
        cv2.imshow("EndoscopeImage", img)
        cv2.waitKey(1)

    def compute_r(self, contour):
        # Compute the radius of the circle that encloses the contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        return radius

    def convert_coordinates(self, x, y, W, H):
        """Convert the coordinates from the default OpenCV frame to the described frame."""
        # Translate the origin to the center of the image
        x_centered = x - W / 2
        y_centered = y - H / 2

        # Invert the X and Y axes
        x_new = -x_centered
        y_new = -y_centered
        # x_new = x_centered
        # y_new = y_centered

        return x_new, y_new

    def detect_green_objects(self, image):
        self.objects = []
        self.tips = []
        # Convert to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Adjust the range for green color
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([90, 255, 255])

        # Threshold the image to keep only the green pixels
        mask = cv2.inRange(hsv_image, lower_green, upper_green)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area and keep the largest four
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

        msgPub = PoseArray()
        centers = []  # To store centers of ellipses
        for contour in sorted_contours:
            if len(contour) >= 5:  # cv2.fitEllipse requires at least 5 points
                ellipse = cv2.fitEllipse(contour)
                center, axes, angle = ellipse

                # Extract the smaller radius (it's half of the smaller axis)
                smaller_radius = min(axes) / 2.0

                # Create SurgTip object
                surg_tip = SurgTip(center[0], center[1], smaller_radius, gamma=0.25)

                centers.append(center)
                self.tips.append(surg_tip)
                cv2.ellipse(
                    image, ellipse, (0, 0, 255), 2
                )  # Drawing in red color for visibility

                msg = Pose()
                msg.position.x = center[0]
                msg.position.y = center[1]
                msg.position.z = int(smaller_radius)
                msgPub.poses.append(msg)

        for tip in self.tips:
            tip.gamma = 1 / len(self.tips)

        s_track, r_track = self.find_center_radius_for_roi(self.tips)
        cv2.circle(
            image,
            (int(s_track[0]), int(s_track[1])),
            int(r_track),
            (
                0,
                255,
            ),
            2,
        )
        # add circle position to the image
        cv2.circle(
            image,
            (int(s_track[0]), (int(s_track[1]))),
            5,
            (
                0,
                255,
            ),
            2,
        )
        tx, ty = self.convert_coordinates(s_track[0], s_track[1], 2048, 1088)
        msg = Point()
        msg.x = tx
        msg.y = ty
        msg.z = 0
        self.pub_green.publish(msg)
        self.tools_pub_.publish(msgPub)

        return image

    def find_center_radius_for_roi(self, tips):
        """
        Find the center to track s_k based on the weighted sum of centers s_i.

        :param objects: List of SurgTip objects with x, y coordinates, radius r, and gamma
        :return: Tuple containing the center to track (s_k_x, s_k_y)
        """
        s_k_x = 0
        s_k_y = 0

        for obj in tips:
            s_k_x += obj.gamma * obj.x
            s_k_y += obj.gamma * obj.y

        #  calculate the radius of the ROI
        # find the max distance from s_k to any s_i and r_i use numpy to do this quickly
        r_i = np.array([obj.r for obj in tips])
        s_i = np.array([[obj.x, obj.y] for obj in tips])
        s_k = np.array([s_k_x, s_k_y])
        r_k = np.max(np.linalg.norm(s_i - s_k, axis=1) + r_i)
        roi_radius = r_k

        return (s_k_x, s_k_y), roi_radius


def main():
    rospy.init_node("green_object_aligner")
    visual_servo = ActiveInferenceVisualServo()

    rospy.spin()


if __name__ == "__main__":
    main()

# last used joint values.
# J:-0.266338 0.0593076   1.79344   3.57207 -0.470811   5.56418
