import rospy
import cv2
import threading
import numpy as np
from collections import defaultdict, deque
from scipy.optimize import minimize
from sensor_msgs.msg import Image as ImageMsg
from geometry_msgs.msg import PoseArray, Point
from cv_bridge import CvBridge, CvBridgeError
import copy


class SurgTip:
    def __init__(self, x, y, r, gamma=1.0):
        self.x, self.y, self.r, self.gamma = x, y, r, gamma


def estimate_velocity(points, dt):
    return np.diff(points, axis=0) / dt


def estimate_acceleration(velocities, time_interval):
    # Using simple finite differences to estimate acceleration
    accelerations = np.diff(velocities, axis=0) / time_interval
    return accelerations


def estimate_roi_inv_pend(
    cluster_centers, speeds, initial_roi_center, dt=0.5, alpha=0.8, gain=0.8
):
    mass, delta_x = 1, 0
    speeds_np = np.array(speeds)
    accelerations = estimate_acceleration(speeds_np, dt)
    # damped_speeds = speeds_np[:-1] * (1 - gain * np.abs(accelerations))
    damped_speeds = speeds_np

    def objective_function_inv_pend(roi_center):
        nonlocal delta_x
        new_r_vectors = np.array(cluster_centers[:-1]) - roi_center
        min_length = min(new_r_vectors.shape[0], damped_speeds.shape[0])

        net_external_torque = np.sum(
            new_r_vectors[:min_length, 0] * damped_speeds[:min_length, 1]
            - new_r_vectors[:min_length, 1] * damped_speeds[:min_length, 0]
        )
        tau_eq = -net_external_torque
        F_net = np.sum(mass * np.linalg.norm(damped_speeds, axis=1))
        delta_x = tau_eq / F_net if F_net != 0 else 0
        return np.abs(tau_eq)

    result = minimize(objective_function_inv_pend, initial_roi_center)
    new_center = alpha * np.array(initial_roi_center) + (1 - alpha) * (
        result.x + delta_x
    )
    return new_center, delta_x


def estimate_roi_net_torque_with_accel(
    cluster_centers, accelerations, initial_roi_center
):
    def objective_function_net_torque(roi_center):
        new_r_vectors = np.array(cluster_centers) - np.array(roi_center)
        accelerations_2d = np.array(accelerations)
        min_length = min(new_r_vectors.shape[0], accelerations_2d.shape[0])
        net_torque = np.sum(
            new_r_vectors[min_length:, 0] * accelerations_2d[min_length:, 1]
            - new_r_vectors[min_length:, 1] * accelerations_2d[min_length:, 0]
        )
        return np.abs(net_torque)

    result_net_torque = minimize(objective_function_net_torque, initial_roi_center)
    new_center_net_torque = result_net_torque.x
    return new_center_net_torque


class RoiProcessor:
    def __init__(self):
        self.image_sub = rospy.Subscriber(
            "/ximea_cam/image_raw", ImageMsg, self.image_callback
        )
        self.tools_sub = rospy.Subscriber(
            "tracked_tools", PoseArray, self.tools_callback
        )
        self.image_pub = rospy.Publisher("/visualized_image", ImageMsg, queue_size=1)
        self.roi_pub_dynamic_ = rospy.Publisher(
            "/dynamic_roi_center", Point, queue_size=1
        )
        self.roi_pub_static_ = rospy.Publisher(
            "/static_roi_center", Point, queue_size=1
        )

        self.bridge = CvBridge()
        self.multi_tips = defaultdict(deque)
        self.tips = []
        self.latest_image = None
        self.dynamic_center = None
        self.static_center = None
        self.lock = threading.Lock()

        self.processing_thread = threading.Thread(target=self.draw_roi)
        self.processing_thread.start()

        self.visualization_thread = threading.Thread(target=self.visualize_image)
        self.visualization_thread.start()
        rospy.sleep(1.0)
        rospy.loginfo("RoiProcessor initialized")

    def tools_callback(self, msg):
        with self.lock:

            tips = [None] * len(msg.poses)  # Pre-allocate list with None
            for i, p in enumerate(msg.poses):
                if p.orientation.x > 0:  # Tool is detected
                    tip = SurgTip(p.position.x, p.position.y, p.position.z, 1.0)
                    tips[i] = tip  # Place the tip at the corresponding index

                    if i not in self.multi_tips:
                        self.multi_tips[i] = deque(maxlen=50)
                    self.multi_tips[i].append(tip)
                else :
                    tips[i] = None
                    # print("not detected i = ", i)

            self.tips = tips  # Directly assign, keeping None for undetected tools

    def get_gamma_based_on_mask_radius(self, tips):
        # Calculate the sum of all radii
        updated_tips = []
        sum_of_radii = sum(t.r for t in tips)
        # print("----------------")
        # print("sum = ", sum_of_radii)
        # Calculate gamma for each tip based on its radius
        for t in tips:
            gamma = t.r / sum_of_radii
            
            t.gamma = gamma
            # print("t.gamma = ", t.gamma)
            updated_tips.append(t)
        # print("-------------------------")
        return updated_tips

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

    def draw_roi(self):
        while True:
            with self.lock:
                # Initialize counter for detected tips
                num_detected_tips = 0

                # Single loop to both count detected tips and set gamma
                valid_tips = []
                for tip in self.tips:
                    if tip is not None:
                        num_detected_tips += 1
                        valid_tips.append(tip)

                if num_detected_tips == 0:
                    rospy.sleep(0.01)
                    continue
                # Calculate gamma_value once and apply to all tips
                gamma_value = 1 / num_detected_tips
                
                for tip in valid_tips:
                    tip.gamma = gamma_value

                valid_tips = self.get_gamma_based_on_mask_radius(valid_tips)
                
                s_track, r_track = self.find_center_radius_for_roi(valid_tips)
                r_track = int(r_track)
                self.static_center = s_track, r_track

                # print("valid tips :", len(valid_tips))
                # print("num_detected_tip = ", num_detected_tips)
                tx, ty = self.convert_coordinates(s_track[0], s_track[1], 2048, 1088)
                static_msg = Point()
                static_msg.x = tx
                static_msg.y = ty
                static_msg.z = r_track
                self.roi_pub_static_.publish(static_msg)

                if all(len(tips) >= 10 for tips in self.multi_tips.values()):
                    speeds, accelerations, cluster_centers = [], [], []
                    combined_samples = defaultdict(list)

                    for channel, samples in self.multi_tips.items():
                        for i, tip in enumerate(samples):
                            combined_samples[i].append(tip)

                    for samples in combined_samples.values():
                        p_center, v, a = self.gaussian_fit_to_motion(samples, 1.0)
                        cluster_centers.append(p_center)
                        speeds.append(v)
                        accelerations.append(a)

                    new_center_inv_pend, delta_x = estimate_roi_inv_pend(cluster_centers, speeds, [s_track[0], s_track[1]])
                    d_r = int(r_track + delta_x)
                    self.dynamic_center = new_center_inv_pend, d_r
                    
                    d_x, d_y = self.convert_coordinates(new_center_inv_pend[0], new_center_inv_pend[1], 2048, 1088)
                    roi_msg = Point()
                    roi_msg.x = d_x
                    roi_msg.y = d_y
                    roi_msg.z = d_r
                    self.roi_pub_dynamic_.publish(roi_msg)

            rospy.sleep(1.0 / 15.0)

    def visualize_image(self):
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
        while True:
            if self.latest_image is not None:
                with self.lock:
                    vis_image = self.latest_image.copy()
                    for i, tip in enumerate(self.tips):
                        if tip is not None:  # Check if the tip is detected
                            color = colors[i % 4]  # Assign a fixed color based on index
                            cv2.circle(
                                vis_image,
                                (int(tip.x), int(tip.y)),
                                int(tip.r),
                                color,
                                2,
                            )

                    if self.static_center is not None:  # Check added here
                        st_center, st_radius = self.static_center
                        cv2.circle(
                            vis_image,
                            (int(st_center[0]), int(st_center[1])),
                            int(st_radius),
                            (0, 255, 0),
                            2,
                        )

                    # if self.dynamic_center is not None:  # Check added here
                    #     dyn_center, dyn_radius = self.dynamic_center
                    #     cv2.circle(
                    #         vis_image,
                    #         (
                    #             int(dyn_center[0]),
                    #             int(dyn_center[1]),
                    #         ),  # Corrected this line
                    #         int(dyn_radius),
                    #         (0, 0, 255),
                    #         2,
                    #     )

                try:
                    self.image_pub.publish(self.bridge.cv2_to_imgmsg(vis_image, "bgr8"))
                    self.static_center = None
                    self.dynamic_center = None
                except CvBridgeError as e:
                    print(e)

            rospy.sleep(1.0 / 15.0)

    def gaussian_fit_to_motion(self, samples, time_interval):
        numerical_samples = [[tip.x, tip.y] for tip in samples]

        samples_np = np.array(numerical_samples)

        mean_position = np.mean(samples_np, axis=0)

        velocities = estimate_velocity(numerical_samples, time_interval)

        accelerations = estimate_acceleration(velocities, time_interval)

        # Use vector sum for velocity and acceleration instead of mean

        vector_sum_velocity = np.sum(velocities, axis=0)

        vector_sum_acceleration = np.sum(accelerations, axis=0)

        var_velocity = np.var(velocities, axis=0)

        # Use vector sum in the condition

        # if np.all(var_velocity < np.abs(vector_sum_velocity)):
        #     return mean_position, np.array([0, 0]), np.array([0, 0])

        return mean_position, vector_sum_velocity, vector_sum_acceleration

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
        # print("s_k =", s_k )
        # print("radius = ", roi_radius)
        # print("r_i = ", r_i)
        # print("s_i = ", s_i)
        return (s_k_x, s_k_y), roi_radius

    def convert_coordinates(self, x, y, W, H):
        """Convert the coordinates from the default OpenCV frame to the described frame."""
        # Translate the origin to the center of the image
        x_centered = x - W / 2
        y_centered = y - H / 2

        # Invert the X and Y axes
        x_new = -x_centered
        y_new = -y_centered

        return x_new, y_new


if __name__ == "__main__":
    rospy.init_node("roi_processor_node", anonymous=False)

    # Initialize RoiProcessor
    roi_processor = RoiProcessor()

    rospy.spin()
