import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from scipy.optimize import minimize
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Pose, Twist, Point
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
import AIP
from scipy.stats import entropy
from scipy.special import softmax
import time


def extract_action_probabilities(B_matrix, action):
    return B_matrix[:, :, action]


def calculate_entropy_based_confidence(B_matrix, k=3.0):

    action_confidence_scores = {}
    max_possible_entropy = entropy([0.2, 0.2, 0.2, 0.2, 0.2], base=2)
    scores = {}
    for action in range(4):  # 4 actions
        action_probabilities = extract_action_probabilities(B_matrix, action)
        confidence_score = 0
        most_likely_state = None
        highest_prob = 0.0

        for state in range(action_probabilities.shape[1]):
            max_prob_state = np.argmax(action_probabilities[:, state])
            max_prob_value = action_probabilities[max_prob_state, state]

            if max_prob_value > highest_prob:

                highest_prob = max_prob_value

                most_likely_state = max_prob_state
                # Find most likely state for this action

                # most_likely_state = np.argmax(np.sum(action_probabilities, axis=0))

                # print("a dist ~ ", action_probabilities[:,most_likely_state])
                column_entropy = entropy(
                    action_probabilities[:, most_likely_state], base=2
                )
                # print("h~ ", column_entropy)
                confidence_score = 1 / (
                    1 + np.exp(-k * (max_possible_entropy - column_entropy))
                )

                scores[most_likely_state] = confidence_score
        # rospy.sleep(2.0)
        action_confidence_scores[f"Action {action + 1}"] = (
            round(scores[most_likely_state], 2),
            f"State {most_likely_state + 1}",
            round(highest_prob, 2),
        )

    return action_confidence_scores


# Function to Check Confidence Scores
def check_confidence_scores(action_scores, theta):
    for action, (confidence, _, _) in action_scores.items():
        if confidence < theta:
            return False
    return True


#  Function to Map Actions to States
def map_actions_to_states(action_scores, action_names, state_names):
    action_state_map = {}
    for i, (action, (_, state, _)) in enumerate(action_scores.items()):
        action_state_map[action_names[i]] = state_names[int(state.split(" ")[1]) - 1]
    return action_state_map


#  Function to Determine State based on Delta_x and Delta_y
def determine_state(current_position, previous_state):
    delta_x = current_position[0] - previous_state[0]
    delta_y = current_position[1] - previous_state[1]

    if delta_x == 0 and delta_y == 0:
        return 4
    if abs(delta_x) > abs(delta_y):
        return 0 if delta_x > 0 else 1
    else:
        return 2 if delta_y > 0 else 3


def get_opposite_state(state, state_names):

    if state == "moved_right":

        return "moved_left"

    elif state == "moved_left":

        return "moved_right"

    elif state == "moved_up":

        return "moved_down"

    elif state == "moved_down":

        return "moved_up"

    else:

        return "did_not_move"


class GaussianDistribution:
    """A simple class to represent a 2D Gaussian distribution with mean and covariance."""

    def __init__(self, mean, covariance):
        self.mean = np.array(mean)
        self.covariance = np.array(covariance)

    def update(self, observation, observation_covariance):
        """Bayesian update of the Gaussian distribution given a new observation."""
        kalman_gain = np.dot(
            self.covariance, np.linalg.inv(self.covariance + observation_covariance)
        )
        self.mean = self.mean + np.dot(kalman_gain, (observation - self.mean))
        self.covariance = self.covariance - np.dot(kalman_gain, self.covariance)


class MDPStructure:
    def __init__(self, name):
        self.name = name  # Name of this specific MDP structure

        self.V = np.array(
            [0, 1, 2, 3]
        )  # Allowable policies, it indicates policies of depth 1
        # Number of states: Moved Up, Moved Down, Moved Left, Moved Right, Did not move
        self.n_states = 5
        # Number of states: Moved Up, Moved Down, Moved Left, Moved Right
        # self.n_states = 4

        # Number of actions: Move Up, Move Down, Move Left, Move Right
        self.n_actions = 4

        # Number of sensory inputs, same as states
        self.n_outcomes = self.n_states

        # Variational Bayes iterations
        self.n_iter = 4  # Just an example, should be tailored to specific needs

        # # Additional states (from comments, exact purpose is not clear)
        # self.n_states_s = 2  # Just an example, should be tailored to specific needs

        # Allowable actions initiation (Transition matrices)
        self.B = np.zeros((self.n_states, self.n_states, self.n_actions))

        # dominant_prob = 0.7
        # other_prob = (1 - dominant_prob) / (self.n_states - 1)
        # for action in range(self.n_actions):
        #     # Set dominant effect (diagonal elements)
        #     self.B[action, action, :] = dominant_prob
        #     # Set other effects
        #     for state in range(self.n_states):
        #         if state != action:
        #             self.B[state, action, :] = other_prob
        uniform_prob = 1 / self.n_states
        self.B = np.full((self.n_states, self.n_states, self.n_actions), uniform_prob)

        # Likelihood matrix
        self.A = np.eye(self.n_states)  # Identity mapping

        # Prior preferences (initially set to zero, so no preference)
        self.C = np.zeros((self.n_states, 1))

        # Belief about initial state all states are equally likely
        # self.D = np.array([[0.25], [0.25], [0.25], [0.25]])
        # self.D = np.array([[0.1], [0.1], [0.1], [0.1], [0.6]])
        self.D = np.array([[0.2], [0.2], [0.2], [0.2], [0.2]])

        # Initial guess about the states (this is updated over time)
        # self.d = np.array([[0.25], [0.25], [0.25], [0.25]])
        self.d = np.array([[0.2], [0.2], [0.2], [0.2], [0.2]])

        # For example, setting self.E = np.array([[1.1], [1.0], [1.0], [1.0]]) would slightly favor the "Move Up" action.
        # Preference about actions (move up should prefere move up, etc.)
        self.E = np.zeros((self.n_actions, 1))

        # Learning rate for initial state update
        self.kappa_d = 0.3

        self.action_names = ["move_up", "move_down", "move_left", "move_right"]

        self.count_matrix = np.zeros((self.n_states, self.n_states, self.n_actions))

        self.state_names = [
            "moved_right",
            "moved_left",
            "moved_up",
            "moved_down",
            "did_not_move",
        ]

        # self.state_names = ["moved_right", "moved_left", "moved_up", "moved_down"]

        self.t_horizon = 4
        self.s_init = 4
        self.store_free_energy = {}

    def update_preferences(self, new_pref):
        self.E = new_pref

    def set_default_preferences(self):
        self.E = np.array([[1.0], [1.0], [1.0], [1.0], [1.0]])

    def update_initial_state(self, new_D):
        self.D = new_D

    def update_belief_initial_state(self, new_d):
        self.d = new_d

    def update_prior(self, new_C):
        self.C = new_C

    #  a function to get an action name from an action index:
    def get_action_name(self, action_index):
        return self.action_names[action_index]

    def get_state_name(self, state_index):
        return self.state_names[state_index]

    # Define a function to update the B matrix in the MDP object
    def update_B_matrix(self, observation, action, alpha=0.2):
        # states_num = np.shape(self.B)[0]
        # print("states_num = ", states_num)
        # actions_num  = np.shape(self.B)[2]
        # print("actions_num = ", actions_num)
        # # rospy.sleep(3.0)
        # print("Before update B matrix: ", self.B)

        last_action = action
        last_state = self.s_init
        last_observation = observation

        # Bayesian or weighted average update
        self.B[last_observation, last_state, last_action] = (1 - alpha) * self.B[
            last_observation, last_state, last_action
        ] + alpha

        # Re-normalize only the column that was updated
        self.B[:, last_state, last_action] /= np.sum(self.B[:, last_state, last_action])
        # print("After update B matrix: ", self.B)
        self.s_init = observation
        # rospy.sleep(3.0)

    # compute entropy in for B matrix:
    def compute_entropy(self):
        entropy = 0
        for action in range(self.n_actions):
            for state in range(self.n_states):
                for obs in range(self.n_outcomes):
                    entropy += self.B[obs, state, action] * np.log(
                        self.B[obs, state, action]
                    )
        return -entropy

        # print("After update B matrix: ", self.B)

    # def update_belief_and_state_from_observation(self, obs):
    #     """
    #     Update the belief and state of the MDP based on the last observation.

    #     Parameters:
    #         obs (int): The last observation.

    #     Returns:
    #         mdp (object): The updated MDP object with the modified belief and state.
    #     """
    #     # print before and after for debugging
    #     print("Before update belief: ", self.d)
    #     print("Before update state: ", self.s)
    #     # Step 1: Update the belief
    #     self.d = self.d * self.A[:, obs] / np.sum(self.d * self.A[:, obs])

    #     # Step 2: Update the state
    #     self.s = np.argmax(self.d)

    #     print("After update: ", self.d)
    #     print("After update state: ", self.s)
    #     return self


class DetectedObject:
    def __init__(self, center_x, center_y, width, height):
        self.mean = np.array([center_x, center_y])
        variance_x = (width / 6) ** 2  # Dividing by 6 because of 3-sigma
        variance_y = (height / 6) ** 2
        self.covariance = np.array([[variance_x, 0], [0, variance_y]])
        self.distribution = multivariate_normal(mean=self.mean, cov=self.covariance)
        self.x = center_x
        self.y = center_y


class ActiveInferenceVisualServo:
    def __init__(self):

        self.last_position = (
            None  # Store the last position of the green object for model updates
        )
        self.service = rospy.Service(
            "/trigger_active_inference", Empty, self.service_callback
        )
        self.pub_key_twist0 = rospy.Publisher(
            "keyboard_control/twist_cmd0", Twist, queue_size=1
        )

        self.object_green = None
        self.object_red = None
        # self.image_subscriber = rospy.Subscriber('/ximea_cam/image_raw', Image, self.image_callback)
        self.image_subscriber = rospy.Subscriber("/image", Image, self.image_callback)
        self.camera_mdp = MDPStructure("vs")
        self.pub_green = rospy.Publisher("/green_object", Point, queue_size=1)

    def service_callback(self, request):
        # This is the ROS service callback that gets triggered when the service is called.
        # It will run the active inference control method and then return an empty response.
        print("request received")
        self.active_inference_control()
        return EmptyResponse()

    def apply_camera_action(self, action):
        # This method will apply the chosen action to the camera.
        twist0 = Twist()

        if action == "move_up":  # Move in +x
            twist0.linear.x = 1
        elif action == "move_down":  # Move in -x
            twist0.linear.x = -1
        elif action == "move_right":  # Move in +y
            twist0.linear.y = 1
        elif action == "move_left":  # Move in -y
            twist0.linear.y = -1

        self.pub_key_twist0.publish(twist0)
        # rospy.sleep(1.0)
        rospy.sleep(0.1)
        time.sleep(0.1)

    def apply_actions_to_achieve_target(
        self, target_position, action_state_map, max_error=50
    ):

        while True:
            # self.object_green = self.detect_green_object_center(self.current_image)
            current_position = [self.object_green.x, self.object_green.y]
            # current_position = [self.object_red.x, self.object_red.y]
            error = np.linalg.norm(
                np.array(current_position) - np.array(target_position)
            )
            if error < max_error:
                break

            error_vector = np.array(target_position) - np.array(current_position)

            print("error = ", error_vector)

            # Determine the desired state based on the error vector

            if abs(error_vector[0]) > abs(error_vector[1]):

                desired_state = "moved_right" if error_vector[0] > 0 else "moved_left"

            else:

                desired_state = "moved_up" if error_vector[1] > 0 else "moved_down"

            print("desired _state = ", desired_state)
            # Find the action corresponding to the desired state

            action_to_take = [
                action for action, st in action_state_map.items() if st == desired_state
            ]
            time.sleep(0.1)
            # Check if we found an action to take
            if not action_to_take:
                print("no actions?")
                print("action_state_map = ", action_state_map)
                rospy.sleep(3.0)
                continue

            action_to_take = action_to_take[0]

            self.apply_camera_action(action_to_take)
            time.sleep(0.1)

    def get_observation(self, previous_state):
        current_position = np.array([self.object_green.x, self.object_green.y])
        # current_position = np.array([self.object_red.x, self.object_red.y])
        # print("current position: ", current_position)
        if previous_state is None:
            raise ValueError("Previous state is None. Cannot compute observation.")
        else:
            #  get the biggest change in x and y and return the corresponding observation
            delta_x = current_position[0] - previous_state[0]
            delta_y = current_position[1] - previous_state[1]
            print("delta x: ", delta_x)
            print("delta y: ", delta_y)
            if delta_x == 0 and delta_y == 0:
                return 4
            if abs(delta_x) > abs(delta_y):
                if delta_x > 0:
                    return 0
                else:
                    return 1
            else:
                if delta_y > 0:
                    return 2
                else:
                    return 3

    def active_inference_control(self):

        self.action_names = ["move_up", "move_down", "move_left", "move_right"]

        self.state_names = [
            "moved_right",
            "moved_left",
            "moved_up",
            "moved_down",
            "did_not_move",
        ]

        # target = [640,360]
        target = [632, 544]
        while True:
            init_position = np.array([self.object_green.x, self.object_green.y])
            # init_position = np.array([self.object_red.x, self.object_red.y])
            # print("init position: ", init_position)
            mdp_updated = AIP.aip_select_action(self.camera_mdp)
            selected_a = int(mdp_updated.u[0][0])
            # print("selected action: ", selected_a)
            a_name = self.camera_mdp.get_action_name(selected_a)
            print("action name: ", a_name, "action ", selected_a + 1)
            self.apply_camera_action(a_name)

            observation = self.get_observation(init_position)
            print("state: ", observation + 1)
            print("observation name: ", self.camera_mdp.get_state_name(observation))

            # self.camera_mdp = self.camera_mdp.update_belief_and_state_from_observation(observation)

            self.camera_mdp.update_B_matrix(observation, selected_a)
            # action_to_state = identify_transitions_with_prob_corrected(self.camera_mdp.B)
            # print("action to state: ", action_to_state)
            # vfe = self.camera_mdp.F[:, self.camera_mdp.t_horizon-1]
            action_confidence_scores = calculate_entropy_based_confidence(
                self.camera_mdp.B
            )
            print("action confidence scores: ", action_confidence_scores)
            print("-----------------------------------------")
            if check_confidence_scores(action_confidence_scores, 0.74):

                action_to_states = map_actions_to_states(
                    action_confidence_scores, self.action_names, self.state_names
                )
                print("action to states ~ ", action_to_states)
                break
        print("before function?")
        self.apply_actions_to_achieve_target(target, action_to_states, max_error=50)

        # print(relative_improvements)
        return

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

    def image_callback(self, image_msg):
        # Convert ROS image message to OpenCV image
        self.current_image = CvBridge().imgmsg_to_cv2(
            image_msg, desired_encoding="bgr8"
        )
        # self.current_image = cv2.resize(self.current_image, (1080, 1920))
        rows, cols, _ = self.current_image.shape
        # M = cv2.getRotationMatrix2D((cols/2, rows/2), -90, 1)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -180, 1)
        self.current_image = cv2.warpAffine(self.current_image, M, (cols, rows))
        self.current_image = cv2.flip(self.current_image, 1)
        # self.current_image = cv2.flip(self.current_image, -1)  # '-1' indicates both horizontal and vertical flip
        # Detect the green object's center and create an object instance if valid
        self.object_green = self.detect_green_object_center(self.current_image)
        if self.object_green is not None:
            x, y = self.object_green.x, self.object_green.y
            # assuming that center of the image is the origin
            # tx, ty = self.convert_coordinates(x, y, 2048, 1088)
            tx, ty = self.convert_coordinates(x, y, 1080, 1080)
            print(
                "green object assuming that center of the image is the origin x = ",
                tx,
                " y = ",
                ty,
            )
            msg = Point()
            msg.x = tx
            msg.y = ty
            self.pub_green.publish(msg)
        # print("self.object_green x= ", self.object_green.x, " y = ", self.object_green.y)
        # self.last_position = np.array([self.object_green.x, self.object_green.y])
        # Detect the red object's center and create an object instance if valid
        # self.object_red = self.detect_red_object_center(self.current_image)
        # Convert to grayscale and apply threshold
        # gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        # _, thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

        # # Find contours
        # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # Draw bounding box around the largest contour (assuming it's the ROI)
        # largest_contour = max(contours, key=cv2.contourArea)
        # x, y, w, h = cv2.boundingRect(largest_contour)
        # cv2.rectangle(self.current_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # # Add grid (rule of thirds)
        grid_color = (255, 255, 255)  # White color
        thickness = 1

        # # Vertical lines
        cv2.line(
            self.current_image, (cols // 3, 0), (cols // 3, rows), grid_color, thickness
        )
        cv2.line(
            self.current_image,
            (2 * cols // 3, 0),
            (2 * cols // 3, rows),
            grid_color,
            thickness,
        )
        # Horizontal lines
        cv2.line(
            self.current_image, (0, rows // 3), (cols, rows // 3), grid_color, thickness
        )
        cv2.line(
            self.current_image,
            (0, 2 * rows // 3),
            (cols, 2 * rows // 3),
            grid_color,
            thickness,
        )

        # self.current_image = cv2.resize(self.current_image, (1080, 1080))
        cv2.imshow("EndoscopeImage", self.current_image)
        cv2.waitKey(1)
        # rospy.sleep(0.1)

    def detect_green_object_center(self, image):
        # Convert to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the range for green color
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])

        # Threshold the image to keep only the green pixels
        mask = cv2.inRange(hsv_image, lower_green, upper_green)
        # M = cv2.moments(mask)
        # if M["m00"] != 0:
        #     cX = int(M["m10"] / M["m00"])
        #     cY = int(M["m01"] / M["m00"])

        # else:
        #     cX, cY = 0, 0  # set to some default value if M["m00"] is zero
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Proceed if at least one contour was found
        if contours:
            # Get the largest contour (assuming it's the object of interest)
            largest_contour = max(contours, key=cv2.contourArea)

            # Fit an ellipse to the contour
            ellipse = cv2.fitEllipse(largest_contour)

            # Extract the parameters of the fitted ellipse
            center_x, center_y = map(int, ellipse[0])
            width, height = map(int, ellipse[1])
            # Check the orientation and order the width and height accordingly
            angle = ellipse[2]

            if angle < 90:
                # print("angle green ~ ", angle)
                major_axis, minor_axis = height, width
            else:
                major_axis, minor_axis = width, height

            # Create DetectedObject instance
            object_green = DetectedObject(center_x, center_y, major_axis, minor_axis)
            # Ensure that width and height are valid
            if width > 0 and height > 0:
                eig_vals_green, eig_vecs_green = np.linalg.eigh(object_green.covariance)

                # Calculate the angle and the axes lengths of the ellipse for object_green (3-sigma confidence interval)
                angle_green = np.degrees(
                    np.arctan2(eig_vecs_green[1, 0], eig_vecs_green[0, 0])
                )
                # print("angle_green = ", angle_green)
                axes_length_green = 3 * np.sqrt(
                    eig_vals_green
                )  # 3-sigma confidence interval

                # Draw the object_green as an ellipse with a custom color (200, 200, 200)
                cv2.ellipse(
                    self.current_image,
                    (int(object_green.x), int(object_green.y)),
                    (int(axes_length_green[0]), int(axes_length_green[1])),
                    angle_green,
                    0,
                    360,
                    (200, 200, 200),
                    3,
                )

                # print("have green ellipse")
                return object_green

        return None

    def detect_red_object_center(self, image):
        # Convert to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the range for bright red color (lower half)
        lower_red1 = np.array([0, 150, 150])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)

        # Define the range for bright red color (upper half)
        lower_red2 = np.array([160, 150, 150])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

        # Combine the two masks
        mask = cv2.bitwise_or(mask1, mask2)

        # Find the contours of the binary image
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Proceed if at least one contour was found
        if contours:
            # Get the largest contour (assuming it's the object of interest)
            largest_contour = max(contours, key=cv2.contourArea)

            ellipse = cv2.fitEllipse(largest_contour)

            # Extract the parameters of the fitted ellipse
            center_x, center_y = map(int, ellipse[0])
            width, height = map(int, ellipse[1])
            # Check the orientation and order the width and height accordingly
            angle = ellipse[2]
            if angle < 90:
                # print("angle red =  ", angle)
                major_axis, minor_axis = height, width
            else:
                major_axis, minor_axis = width, height

            # Ensure that width and height are valid
            if width > 0 and height > 0:
                object_red = DetectedObject(center_x, center_y, major_axis, minor_axis)
                cv2.ellipse(self.current_image, ellipse, (150, 100, 200), 3)
                eig_vals_green, eig_vecs_green = np.linalg.eigh(object_red.covariance)

                # Calculate the angle and the axes lengths of the ellipse for object_green (3-sigma confidence interval)
                angle_red = np.degrees(
                    np.arctan2(eig_vecs_green[1, 0], eig_vecs_green[0, 0])
                )
                axes_length_green = 3 * np.sqrt(
                    eig_vals_green
                )  # 3-sigma confidence interval

                # Draw the object_green as an ellipse with a custom color (200, 200, 200)
                cv2.ellipse(
                    self.current_image,
                    (int(object_red.x), int(object_red.y)),
                    (int(axes_length_green[0]), int(axes_length_green[1])),
                    angle_red,
                    0,
                    360,
                    (200, 200, 200),
                    3,
                )
                # print("have red ellipse")
                return object_red

        return None


def main():
    rospy.init_node("green_object_aligner")
    visual_servo = ActiveInferenceVisualServo()

    rospy.spin()


if __name__ == "__main__":
    main()
