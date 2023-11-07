# %%
# parameters
# Vel: 60
# Acc: 1.5
# kMaxLinVelEE = 0.03  # [mm/s]
# kMaxAngVelEE = 0.5  # [rad/s]
# Kr_vt = rospy.get_param("~Kr_vt", 0.05)  #

from casadi import *
import pinocchio as pin
import numpy as np
import os
import time
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point, Twist, Pose, PoseStamped
from scipy.spatial.transform import Rotation as R
import copy

# import hppfcl
# from pinocchio.visualize import RVizVisualizer
import tf
import itertools
import csv
from scipy.spatial.transform import Rotation as R

# from asar_endoscope_control.srv import ikChecker, ikCheckerResponse
from geometry_msgs.msg import Pose, Point, Quaternion

from asar_control.msg import FollowJointTargetAction, FollowJointTargetGoal
import actionlib

global roll_angle
roll_angle = 0


def pose_to_se3(pose):
    """
    Convert geometry_msgs/Pose to pinocchio.SE3.

    Args:
    - pose (geometry_msgs.Pose): Input pose.

    Returns:
    - pin.SE3: Output SE3 transformation.
    """
    # Extract position and orientation from Pose
    p = pose.position
    q = pose.orientation

    # Convert position list to numpy array
    trans = np.array([p.x, p.y, p.z])

    # Convert to Pinocchio SE3
    se3 = pin.SE3(pin.Quaternion(q.w, q.x, q.y, q.z).matrix(), trans)

    return se3


def se3_to_pose(se3):
    """
    Convert pinocchio.SE3 to geometry_msgs/Pose.

    Args:
    - se3 (pin.SE3): Input SE3 transformation.

    Returns:
    - geometry_msgs.Pose: Output pose.
    """
    # Extract rotation and translation from SE3
    rot = se3.rotation
    trans = se3.translation

    # Convert to geometry_msgs.Pose
    quat = pin.Quaternion(rot)
    pose = Pose()
    pose.position = Point(trans[0], trans[1], trans[2])
    pose.orientation = Quaternion(quat.x, quat.y, quat.z, quat.w)

    return pose


cwd = os.getcwd()

# List of available robots
robot_ids = ["asar_endoscope"]

# Use current path to store results
results_path = os.path.join(cwd, "results")

# Name CSV file with timestamp
timestamp = time.strftime("%Y%m%d-%H%M%S")
csv_filename = os.path.join(results_path, "results_" + timestamp + ".csv")

# csv_filename = os.path.join(results_path, "results.csv")

# Initialize CSV file header
# with open(csv_filename, "w", newline="") as csvfile:
#     writer = csv.writer(csvfile, delimiter=",")
#     writer.writerow(
#         [
#             "sol_found",
#             "err_rcm",
#             "err_ee",
#             "err_manip",
#             "B_X_END[x]",
#             "B_X_END[y]",
#             "B_X_END[z]",
#             "q0[0]",
#             "q0[1]",
#             "q0[2]",
#             "q0[3]",
#             "q0[4]",
#             "q0[5]",
#             "q0[6]",
#         ]
#     )


# Get the urdf path of the robot
def get_urdf_path(robot_id):
    if robot_id == "asar_endoscope":
        # urdf_path = os.path.join(
        #     cwd,
        #     "/home/medical/crest_ws/src/asar_description/urdf/asar_endoscope_v2.urdf",
        # )
        urdf_path = (
            "/home/colan/crest_ws/src/asar_description/urdf/asar_endoscope_v2.urdf"
        )
    else:
        raise ValueError("Invalid robot id")
    return urdf_path


# Class to solve the inverse kinematics problem
class HqpEndoscopeController:
    def __init__(self, urdf_path, params):
        # Load urdf model
        self.model = pin.buildModelFromUrdf(urdf_path)
        # self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
        #     urdf_path
        # )

        # Create data from model
        self.data = self.model.createData()
        # self.data, self.collision_data, self.visual_data = pin.createDatas(
        #     self.model, self.collision_model, self.visual_model
        # )

        # Get number of joints
        self.nq = self.model.nq

        # Get number of joints in the arm
        self.n_arm_joints = params["n_arm_joints"]

        # Get number of joints in the RST
        self.n_rst_joints = params["n_rst_joints"]

        # Extract joint limits
        self.q_max_ = self.model.upperPositionLimit
        self.q_min_ = self.model.lowerPositionLimit

        # Get the end-effector id
        self.ee_id = self.model.getFrameId(params["ee_link_name"])

        # Get the pre RCM joint id
        self.pre_rcm_id = self.model.getFrameId(params["pre_rcm_joint_name"])

        # Get the post RCM joint id
        self.post_rcm_id = self.model.getFrameId(params["post_rcm_joint_name"])

        # Get the pre link1 joint id
        self.pre_link1_id = self.model.getFrameId(params["pre_link1_joint_name"])

        # Get the post link1 joint id
        self.post_link1_id = self.model.getFrameId(params["post_link1_joint_name"])

        # Get the pre link2 joint id
        self.pre_link2_id = self.model.getFrameId(params["pre_link2_joint_name"])

        # Get the post link2 joint id
        self.post_link2_id = self.model.getFrameId(params["post_link2_joint_name"])

        # Get the integration time
        self.dT = params["dT"]

        # Get the number of joints
        self.nq = self.model.nq

        # Get if manipulability is used
        self.is_manip_on = params["is_manip_on"]

        # Get RCM position
        self.B_p_Ftrocar = np.array(params["rcm_pos"])
        self.rcm_position = params["rcm_pos"]

        # Get Task coefficients Kt
        self.Kt = params["Kt"]

        # Get Joint-distance coefficients Kd
        self.Kd = params["Kd"]

        # Get slack variable coefficients Kw
        self.Kw = params["Kw"]

        # Get residual coefficients Kr
        self.Kr = params["Kr"]

        # Get the error tolerances
        self.eps_r = params["eps.r"]
        self.eps_e = params["eps.e"]
        self.eps_v = params["eps.v"]
        self.eps_w = params["eps.w"]

        # Task dimensions
        self.m_ee = 6
        self.m_rcm = 1
        self.m_manip = 1
        self.m_coll = 1
        self.m_vt = 2
        self.m_align = 1
        self.m_depth = 1

        self.q_start = None
        self.green_obj = None
        self.object_detected = False

        # Print some info
        print(f"model name: {self.model.name}")
        print("Number of frames: {0:2d}".format(self.model.nframes))
        print("Number of Joints: {0:2d}".format(self.model.njoints))
        # print("Names of Joints: {0:2d}".format(self.model.njoints))
        print("Number of Active Joints: {0:2d}".format(self.model.nq))

        # Initialization of joint states
        self.q0 = np.zeros(self.nq)
        self.q1 = np.zeros(self.nq)

        # Subscribers
        self.sub_arm0_q = rospy.Subscriber(
            "/unit0/arm/sim/joint/state",
            JointState,
            self.getArm0JointStateCb,
            queue_size=1,
        )
        self.sub_rst0_q = rospy.Subscriber(
            "/unit0/forceps/sim/joint/state",
            JointState,
            self.getRST0JointStateCb,
            queue_size=1,
        )

        self.sub_green_obj = rospy.Subscriber(
            "green_object", Point, self.getGreenObjCb, queue_size=1
        )
        # self.sub_green_obj = rospy.Subscriber(
        #     "static_roi_center", Point, self.getGreenObjCb, queue_size=1
        # )
        # self.sub_green_obj = rospy.Subscriber(
        #     "dynamic_roi_center", Point, self.getGreenObjCb, queue_size=1
        # )
        # subsribe to twist commands
        self.sub_key_twist0 = rospy.Subscriber(
            "/keyboard_control/twist_cmd0", Twist, self.getTwistCb, queue_size=1
        )

        # Publishers
        self.pub_arm_qd = rospy.Publisher(
            "/unit0/arm/sim/joint/cmd", JointState, queue_size=1
        )

        self.pub_rst_qd = rospy.Publisher(
            "/unit0/forceps/sim/joint/cmd", JointState, queue_size=1
        )

        # Create ik solver service:
        # self.ik_service = rospy.Service("/hqp_solver_service", ikChecker, self.getIKCb)

        # Get current end-effector pose
        pin.framesForwardKinematics(self.model, self.data, self.q0)
        self.B_X_END = copy.deepcopy(self.data.oMf[self.ee_id])
        # print("EE pose frame:", self.B_X_END)
        # print("Camera pose frame:", self.data.oMf[self.pre_rcm_id])
        # print("Tip pose frame", self.data.oMf[self.post_rcm_id])

        # Initialization flags
        self.is_arm0_q_init = False
        self.is_rst0_q_init = False

        self.qpsolver = self.generate_qpsolver()
        self.qpsolver_operational = self.generate_qpsolver_operational()
        self.ux = np.array([-1, 0, 0])
        self.uy = np.array([0, 1, 0])
        # self.saved = False

    def publish_qd(self, qd):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = self.model.names[1 : self.n_arm_joints + 1]
        msg.position = qd[: self.n_arm_joints]
        msg.velocity = np.zeros(self.n_arm_joints)
        msg.effort = np.zeros(self.n_arm_joints)

        self.pub_arm_qd.publish(msg)

    def getArm0JointStateCb(self, msg):
        self.q0[: self.n_arm_joints] = np.array(msg.position)

        if self.is_rst0_q_init == True and controller_ready == False:
            # Get current end-effector pose
            pin.framesForwardKinematics(self.model, self.data, self.q0)
            self.B_X_END = copy.deepcopy(self.data.oMf[self.ee_id])
            # Print EE pose
            # print("EE position:", self.B_X_END.translation)
            # print("EE orientation:", self.B_X_END.rotation)

        if not self.is_arm0_q_init:
            self.is_arm0_q_init = True

    def getRST0JointStateCb(self, msg):
        self.q0[self.n_arm_joints :] = np.array(msg.position[: self.n_rst_joints])
        # self.qd = np.array(msg.velocity)
        # print("q: ", self.q)

        if self.is_arm0_q_init == True and controller_ready == False:
            # Get current end-effector pose
            pin.framesForwardKinematics(self.model, self.data, self.q0)
            self.B_X_END = copy.deepcopy(self.data.oMf[self.ee_id])
            # Print EE pose
            # print("EE position:", self.B_X_END.translation)
            # print("EE orientation:", self.B_X_END.rotation)

        if not self.is_rst0_q_init:
            self.is_rst0_q_init = True

    def generate_qpsolver(self):
        # Optimization variable
        x1 = SX.sym("x", 3 * self.nq, 1)

        # Parameters
        par1 = SX.sym("par", 3 * self.nq, 5 * self.nq + 2)

        Q1 = par1[:, : 3 * self.nq]
        p1 = par1[:, 3 * self.nq]
        C1 = par1[:, 3 * self.nq + 1 : 5 * self.nq + 1].T
        d1 = par1[0 : 2 * self.nq, par1.shape[1] - 1]

        s_opts = {
            "verbose": False,
            "osqp.max_iter": 1000,
            "osqp.verbose": False,
            # "jit": True,
            # "printLevel": "none",  # qpoases
            "warm_start_primal": True,  # osqp
        }

        qp1 = {
            "x": x1,
            "f": 0.5 * x1.T @ Q1 @ x1 + p1.T @ x1,
            "g": d1 - C1 @ x1,
            "p": par1,
        }

        return qpsol("qpsol1", "osqp", qp1, s_opts)

    def generate_qpsolver_operational(self):
        n_dof = 4
        # Optimization variable
        x1 = SX.sym("x", 3 * n_dof, 1)

        # Parameters
        par1 = SX.sym("par", 3 * n_dof, 5 * n_dof + 2)

        Q1 = par1[:, : 3 * n_dof]
        p1 = par1[:, 3 * n_dof]
        C1 = par1[:, 3 * n_dof + 1 : 5 * n_dof + 1].T
        d1 = par1[0 : 2 * n_dof, par1.shape[1] - 1]

        s_opts = {
            "verbose": False,
            "osqp.max_iter": 1000,
            "osqp.verbose": False,
            # "jit": True,
            # "printLevel": "none",  # qpoases
            "warm_start_primal": True,  # osqp
        }

        qp1 = {
            "x": x1,
            "f": 0.5 * x1.T @ Q1 @ x1 + p1.T @ x1,
            "g": d1 - C1 @ x1,
            "p": par1,
        }

        return qpsol("qpsol_operational", "osqp", qp1, s_opts)

    def update_model(self, q_it):
        pin.framesForwardKinematics(self.model, self.data, q_it)

    def perr(self):
        # pin.framesForwardKinematics(self.model, self.data, q_it)
        p_FeeDes = copy.deepcopy(self.B_X_des_Fee.translation)
        p_FeeAct = copy.deepcopy(self.data.oMf[self.ee_id].translation)

        perr = p_FeeDes - p_FeeAct
        return np.linalg.norm(perr)

    def oerr(self):
        o_FeeDes = R.from_matrix(self.B_X_des_Fee.rotation).as_quat()
        o_FeeAct = R.from_matrix(self.data.oMf[self.ee_id].rotation).as_quat()
        # o_FeeAct_conj = np.array(
        #     [o_FeeAct[0], -o_FeeAct[1], -o_FeeAct[2], -o_FeeAct[3]]
        # )
        o_FeeAct_conj = o_FeeAct
        alpha = np.abs(np.dot(o_FeeDes, o_FeeAct_conj))
        oerr = 0 if alpha >= 1 else 2 * np.arccos(alpha)
        return np.linalg.norm(oerr)

    def compute_lambda(self, resolution_x, fov_deg):
        """
        Compute the lambda value for perspective projection.

        Parameters:
        - resolution_x: Width of the image in pixels
        - fov_deg: Field of view in degrees

        Returns:
        - lambda_val: Computed lambda value
        """
        return resolution_x / (2 * np.tan(np.radians(fov_deg) / 2))

    def normalize_rotation(self, rotation):
        r = R.from_matrix(rotation)
        q = r.as_quat()
        q = q / np.linalg.norm(q)
        return R.from_quat(q).as_matrix()

    def project_3D_to_2D(self, targ_3d, Init_cam):
        K = np.array(
            [[2447.10470, 0, 1014.20985], [0, 2422.18638, 471.114582], [0, 0, 1]]
        )
        Rt = np.hstack(
            (Init_cam.rotation, Init_cam.translation)
        )  # Assuming identity rotation and zero translation
        P_homo = np.append(targ_3d, 1)
        p_homo = K @ Rt[:3, :] @ P_homo

        new_pixel_pose = p_homo[:2] / p_homo[2]
        return new_pixel_pose

    def compute_residual_EE_log6(self):
        B_X_act_Fee = copy.deepcopy(self.data.oMf[self.ee_id])
        Fee_X_act_B = pin.SE3(
            B_X_act_Fee.rotation.transpose(),
            np.matmul(
                -B_X_act_Fee.rotation.transpose(),
                B_X_act_Fee.translation,
            ),
        )
        B_X_delta = self.B_X_des_Fee.act(Fee_X_act_B)

        res_ee = pin.log(B_X_delta).vector
        err_ee = np.linalg.norm(res_ee)

        return res_ee, err_ee

    # def compute_expected_change(u_before, v_before, z, twist):
    #     """
    #     Compute the expected change in u and v using the M matrix and the provided twist.

    #     Parameters:
    #     - u_before: Initial u value
    #     - v_before: Initial v value
    #     - z: Depth of the point in the camera's coordinate frame
    #     - twist: Array representing the twist [vx, vy, vz, wx, wy, wz]

    #     Returns:
    #     - expected_change: Expected change in u and v
    #     """

    #     # Define the M matrix using the provided z value
    #     M = np.array([
    #         [lambda_val / z, 0,  -u_before / z, -u_before * v_before /lambda_val,  (lambda_val**2 + u_before**2)/lambda_val, -v_before],
    #         [0, lambda_val / z, -v_before / z,   (-lambda_val**2 - v_before**2) /lambda_val,  u_before*v_before / lambda_val, u_before]
    #     ])

    #     # Compute the expected change using the M matrix and the provided twist
    #     expected_change = np.dot(M, twist)

    #     return expected_change

    def compute_residual_depth(self, d_des):
        l_actual = self.B_X_END.translation - self.rcm_position
        lhat = l_actual / np.linalg.norm(l_actual)
        d_actual = np.linalg.norm(l_actual)
        # res_depth = (d_des - d_actual) * lhat
        res_depth = -(d_des - d_actual)
        err_depth = d_des - d_actual
        print("d_des: ", d_des, "d_actual: ", d_actual, "err_depth: ", err_depth)
        print("d_des: ", d_des, "d_actual: ", d_actual, "err_depth: ", err_depth)
        if abs(res_depth) < 2e-3:
            res_depth = np.array([0])
            err_depth = np.array([0])

        return res_depth, err_depth, lhat

    def compute_residual_VT(self):
        # newTarg2d = self.project_3D_to_2D()

        # To center feature in image
        # res_vt = self.s_center - self.s_target
        target = np.array([self.green_obj[1], -self.green_obj[0]])
        # res_vt = -self.green_obj
        res_vt = -target
        err_vt = np.linalg.norm(res_vt)

        # if err_vt < 100:
        if err_vt < 50:
            res_vt = np.array([0, 0])
            err_vt = 0

        return res_vt, err_vt

    def compute_residual_RCM(self):
        B_X_Fprercm = copy.deepcopy(self.data.oMf[self.pre_rcm_id])
        B_X_Fpostrcm = copy.deepcopy(self.data.oMf[self.post_rcm_id])

        # Computing RCM error
        ps = B_X_Fpostrcm.translation - B_X_Fprercm.translation
        pr = self.B_p_Ftrocar - B_X_Fprercm.translation
        ps_hat = ps / np.linalg.norm(ps)

        B_p_Frcm = B_X_Fprercm.translation + np.transpose(pr) @ np.outer(ps_hat, ps_hat)
        pe = self.B_p_Ftrocar - B_p_Frcm

        # res_rcm = -np.dot(pe, pe)
        res_rcm = -np.linalg.norm(pe)
        err_rcm = np.linalg.norm(pe)

        # if err_rcm < 1e-12:
        if err_rcm < 1e-3:
            err_rcm = 1e-12
            res_rcm = -1e-12

        return res_rcm, ps, pr, pe, err_rcm, B_p_Frcm

    def compute_Jacobian_EE(self, q_it):
        B_Jb_END = pin.computeFrameJacobian(
            self.model, self.data, q_it, self.ee_id, pin.WORLD
        )
        return B_Jb_END

    def compute_Jacobian_RCM(self, q_it, ps, pr, pe):
        ps_hat = ps / np.linalg.norm(ps)
        pe_hat = pe / np.linalg.norm(pe)

        B_Jb_Fprercm = pin.computeFrameJacobian(
            self.model, self.data, q_it, self.pre_rcm_id, pin.LOCAL_WORLD_ALIGNED
        )
        B_Jb_Fpostrcm = pin.computeFrameJacobian(
            self.model, self.data, q_it, self.post_rcm_id, pin.LOCAL_WORLD_ALIGNED
        )

        Jb_ps_hat = (
            (1 / np.linalg.norm(ps))
            * (np.eye(3) - np.outer(ps_hat, np.transpose(ps_hat)))
            @ (B_Jb_Fpostrcm - B_Jb_Fprercm)[:3, :]
        )

        B_Jb_Frcm = -np.transpose(pe_hat) @ (
            (np.eye(3) - np.outer(ps_hat, np.transpose(ps_hat))) @ B_Jb_Fprercm[:3, :]
            + (
                np.outer(ps_hat, np.transpose(pr))
                + np.dot(np.transpose(pr), ps_hat) * np.eye(3)
            )
            @ Jb_ps_hat
        )
        B_Jb_Frcm = B_Jb_Frcm.reshape(self.nq, 1)

        return B_Jb_Frcm

    def compute_Jacobian_depth(self, q_it, l_hat):
        B_Jb_CAM = pin.computeFrameJacobian(
            self.model, self.data, q_it, self.pre_rcm_id, pin.LOCAL_WORLD_ALIGNED
        )
        B_Jb_END = pin.computeFrameJacobian(
            self.model, self.data, q_it, self.post_rcm_id, pin.LOCAL_WORLD_ALIGNED
        )
        B_Jb_depth = -np.transpose(l_hat) @ (B_Jb_END - B_Jb_CAM)[:3, :]
        B_Jb_depth = B_Jb_depth.reshape(self.nq, 1)

        return B_Jb_depth

    def compute_Manip_EE(self, q_it):
        Jb = pin.computeFrameJacobian(
            self.model, self.data, q_it, self.ee_id, pin.WORLD
        )
        return np.sqrt(np.linalg.det(Jb.dot(Jb.T)))

    def compute_GradManip_EE(self, q_it):
        # print(q_it)
        grad_mp = np.empty((self.nq, 1))
        mp_dq = 1e-3
        for joint in range(self.nq):
            Ei = np.zeros((self.nq))
            Ei[joint] = 1
            grad_mp[joint] = (
                self.compute_Manip_EE(q_it + mp_dq * Ei)
                - self.compute_Manip_EE(q_it - mp_dq * Ei)
            ) / (2 * mp_dq)
        return grad_mp

    def generate_QP_problem_OperationSpace(
        self,
        priority: int,
        n_tasks: int,
        tasks_dim: list,
        Np: np.ndarray,
        x_opt: np.ndarray,
        At: list,
        bt: list,
        Ct: np.ndarray,
        dt: np.ndarray,
        Kt: list,
        Kd: float,
        Kw: float,
    ):
        n_dof = 4  # Operatonal space dimension
        if priority == 1:
            N = np.eye(n_dof)
            qd_opt = np.zeros((n_dof, 1))
        else:
            N = Np
            qd_opt = x_opt

        m = sum(tasks_dim)

        Ap = np.zeros((m + n_dof, n_dof))
        bp = np.zeros((m + n_dof, 1))

        Abar = np.zeros((m + 3 * n_dof, 3 * n_dof))
        bbar = np.zeros((m + 3 * n_dof, 1))
        Cbar = np.zeros((2 * n_dof, 3 * n_dof))
        dbar = np.zeros((2 * n_dof, 1))

        r_index = 0
        for k in range(n_tasks):
            Ap[r_index : r_index + tasks_dim[k], :] = At[k]
            bp[r_index : r_index + tasks_dim[k], :] = -np.sqrt(Kt[k]) * (
                At[k] @ qd_opt - bt[k]
            )
            r_index += tasks_dim[k]

        Ap[r_index : r_index + n_dof, :] = np.sqrt(Kd) * np.eye(n_dof)
        bp[r_index : r_index + n_dof, :] = np.zeros((n_dof, 1))

        Abar[0 : Ap.shape[0], 0:n_dof] = Ap @ N
        Abar[Ap.shape[0] :, n_dof:] = np.sqrt(Kw) * np.eye(2 * n_dof)

        bbar[0 : bp.shape[0], :] = bp
        bbar[bp.shape[0] :, :] = np.zeros((2 * n_dof, 1))

        Cbar[0 : 2 * n_dof, 0:n_dof] = Ct @ N
        Cbar[0 : 2 * n_dof, n_dof:] = -np.eye(2 * n_dof)

        dbar[0 : 2 * n_dof, :] = dt - Ct @ qd_opt

        Q = np.transpose(Abar) @ Abar
        p = -np.transpose(Abar) @ bbar

        # print(
        #     "Ap (shape):",
        #     Ap.shape,
        #     "| bp (shape):",
        #     bp.shape,
        #     "| Abar (shape):",
        #     Abar.shape,
        #     "| bbar (shape):",
        #     bbar.shape,
        #     "| Cbar (shape):",
        #     Cbar.shape,
        #     "| dbar (shape):",
        #     dbar.shape,
        #     "| Q (shape):",
        #     Q.shape,
        #     "| p (shape):",
        #     p.shape,
        # )

        return Q, p, Cbar, dbar

    def generate_QP_problem(
        self,
        priority: int,
        n_tasks: int,
        tasks_dim: list,
        Np: np.ndarray,
        x_opt: np.ndarray,
        At: list,
        bt: list,
        Ct: np.ndarray,
        dt: np.ndarray,
        Kt: list,
        Kd: float,
        Kw: float,
    ):
        if priority == 1:
            N = np.eye(self.nq)
            qd_opt = np.zeros((self.nq, 1))
        else:
            N = Np
            qd_opt = x_opt

        m = sum(tasks_dim)

        Ap = np.zeros((m + self.nq, self.nq))
        bp = np.zeros((m + self.nq, 1))

        Abar = np.zeros((m + 3 * self.nq, 3 * self.nq))
        bbar = np.zeros((m + 3 * self.nq, 1))
        Cbar = np.zeros((2 * self.nq, 3 * self.nq))
        dbar = np.zeros((2 * self.nq, 1))

        r_index = 0
        for k in range(n_tasks):
            Ap[r_index : r_index + tasks_dim[k], :] = At[k]
            bp[r_index : r_index + tasks_dim[k], :] = -np.sqrt(Kt[k]) * (
                At[k] @ qd_opt - bt[k]
            )
            r_index += tasks_dim[k]

        Ap[r_index : r_index + self.nq, :] = np.sqrt(Kd) * np.eye(self.nq)
        bp[r_index : r_index + self.nq, :] = np.zeros((self.nq, 1))

        Abar[0 : Ap.shape[0], 0 : self.nq] = Ap @ N
        Abar[Ap.shape[0] :, self.nq :] = np.sqrt(Kw) * np.eye(2 * self.nq)

        bbar[0 : bp.shape[0], :] = bp
        bbar[bp.shape[0] :, :] = np.zeros((2 * self.nq, 1))

        Cbar[0 : 2 * self.nq, 0 : self.nq] = Ct @ N
        Cbar[0 : 2 * self.nq, self.nq :] = -np.eye(2 * self.nq)

        dbar[0 : 2 * self.nq, :] = dt - Ct @ qd_opt

        Q = np.transpose(Abar) @ Abar
        p = -np.transpose(Abar) @ bbar

        # print(
        #     "Ap (shape):",
        #     Ap.shape,
        #     "| bp (shape):",
        #     bp.shape,
        #     "| Abar (shape):",
        #     Abar.shape,
        #     "| bbar (shape):",
        #     bbar.shape,
        #     "| Cbar (shape):",
        #     Cbar.shape,
        #     "| dbar (shape):",
        #     dbar.shape,
        #     "| Q (shape):",
        #     Q.shape,
        #     "| p (shape):",
        #     p.shape,
        # )

        return Q, p, Cbar, dbar

    def getProjectionOnIVP(self, ux, uy):
        # Define local basis vectors for the reference plane
        rx = np.array([-1, 0, 0])  # Points to the left of the global x
        ry = np.array([0, 1, 0])  # Points to the front, same as global y
        rz = np.array([0, 0, -1])  # Points into the plane, opposite to global z (up)

        # Store these as rows in a 3x3 NumPy array to represent the reference plane
        reference_plane = np.array([rx, ry, rz])
        normal_vector = np.cross(rx, ry)
        given_ux = ux / np.linalg.norm(ux)
        given_uy = uy / np.linalg.norm(uy)

        # # Compute the projections magnitude
        # projection_ux_on_x = np.dot(given_ux, rx)
        # projection_uy_on_y = np.dot(given_uy, ry)
        # Compute the projection vectors
        projection_ux_on_x = (np.dot(given_ux, rx) / np.dot(rx, rx)) * rx
        projection_uy_on_y = (np.dot(given_uy, ry) / np.dot(ry, ry)) * ry

        # projection_ux_on_x_normalized = projection_ux_on_x / np.linalg.norm(projection_ux_on_x)
        # projection_uy_on_y_normalized = projection_uy_on_y / np.linalg.norm(projection_uy_on_y)

        return projection_ux_on_x, projection_uy_on_y

    def storeAxisProjections(
        self,
    ):
        point = numpy.array(self.B_X_END.translation)
        vz = point - self.rcm_position
        uz = vz / np.linalg.norm(vz)

        uy = -np.cross(uz, [1, 0, 0])
        ux = np.cross(uy, uz)
        px, py = self.getProjectionOnIVP(ux, uy)
        self.ux = px
        self.uy = py
        self.saved = True

    def project_onto_plane(self, v, n):
        return v - np.dot(v, n) * n

    def getIVPConstraints(self, ux, uy):
        # R = np.array([

        #     [-0.999963, 4.211e-08, -0.00862202],

        #     [-0.00423881, 0.870804, 0.491612],

        #     [0.00750811, 0.49163, -0.870772]

        # ])

        # R = np.array(
        #     [
        #         [-9.99989587e-01, 3.62276011e-03, 2.77524857e-03],
        #         [4.51170407e-03, 6.93359148e-01, 7.20578057e-01],
        #         [6.86237458e-04, 7.20583074e-01, -6.93368273e-01],
        #     ]
        # )
        # 0 deg endoscope
        # R = np.array(
        #     [
        #         [-0.98694063, -0.02886373, 0.15847737],
        #         [0.07431554, 0.79127546, 0.60692698],
        #         [-0.14291743, 0.61077822, -0.77879687],
        #     ]
        # )
        #  30 deg endoscope
        # self.alignment_ref = np.array(
        #     [
        #         [-0.99984756, 0.01537409, -0.0082761],
        #         [0.01090215, 0.91996043, 0.39185961],
        #         [0.01363817, 0.39170965, -0.9199878],
        #     ]
        # )

        self.alignment_ref = np.array(
            [
                [-0.01846654, 0.9994546, -0.02737687],
                [0.87401319, 0.0294341, 0.48500988],
                [0.48555117, -0.0149713, -0.87408005],
            ]
        )

        x = self.alignment_ref[:, 0]
        y = self.alignment_ref[:, 1]

        nominator = np.cross(x, ux) + np.cross(y, uy)
        print("nominator: ", nominator)
        nominator_magn = np.linalg.norm(nominator)
        denominator = np.cross(x, uy) - np.cross(y, ux)
        print("denominator: ", denominator)
        denominator_magn = np.linalg.norm(denominator)
        # if denominator == 0:
        #     rospy.logerr("denominator is zero")
        sign = np.sign(np.dot(nominator, denominator))
        print(sign)
        angle = sign * np.arctan(nominator_magn / denominator_magn)
        print("ivp constraint angle = ", angle * 180 / np.pi)
        return angle

    def solveAllignment(self, currentPose, targetPose):
        u_x = currentPose.rotation[:, 0]
        u_y = currentPose.rotation[:, 1]
        theta_current = self.estimateMisallignment(u_x, u_y)
        # Initialize A_theta as zeros
        A_theta = np.zeros((1, self.nq))
        # Set the w_z in A to 1
        # A_theta[0, 6] = 1.0
        b_theta = -0.1 * theta_current

    def p_control_alignment(self, currentPose):
        ux = currentPose.rotation[:, 0]
        uy = currentPose.rotation[:, 1]
        theta = self.estimateMisallignment(ux, uy)

        tolerance = 0.1
        k = 0.1

    def getAlignmentAngle(self, ux1, uy1, ux2, uy2, normal_vector):
        # Project the vectors onto the plane defined by normal_vector
        ux1_proj = self.project_onto_plane(ux1, normal_vector)
        uy1_proj = self.project_onto_plane(uy1, normal_vector)
        ux2_proj = self.project_onto_plane(ux2, normal_vector)
        uy2_proj = self.project_onto_plane(uy2, normal_vector)

        # ux1_proj = ux1
        # uy1_proj = ux2
        # ux2_proj = uy1
        # uy2_proj = uy2

        # Compute the diagonal vectors
        # D1 = ux1_proj + uy1_proj
        D1 = ux1 + uy1

        D2 = ux2_proj + uy2_proj

        # Check if they are parallel by taking their cross product
        cross_product = np.cross(D1, D2)

        # Calculate the magnitudes of D1, D2, and the cross product
        mag_D1 = np.linalg.norm(D1)
        mag_D2 = np.linalg.norm(D2)
        mag_cross_product = np.linalg.norm(cross_product)

        if mag_cross_product == 0:
            print("parallel")
            return 0.0  # The vectors are parallel, no rotation needed

        # Calculate the angle between D1 and D2 in the plane orthogonal to the cross product
        cos_theta = np.dot(D1, D2) / (mag_D1 * mag_D2)
        theta = np.arccos(cos_theta)
        print("theta: ", theta * 180 / np.pi)

        return theta

    def estimateMisallignment(self, X_actual):
        # Given SE3 pose
        # self.alignment_ref = np.array([

        #     [-0.999963, 4.211e-08, -0.00862202],

        #     [-0.00423881, 0.870804, 0.491612],

        #     [0.00750811, 0.49163, -0.870772]

        # ])

        # self.alignment_ref = np.array(
        #     [
        #         [-9.99989587e-01, 3.62276011e-03, 2.77524857e-03],
        #         [4.51170407e-03, 6.93359148e-01, 7.20578057e-01],
        #         [6.86237458e-04, 7.20583074e-01, -6.93368273e-01],
        #     ]
        # )
        # 0 deg endoscope
        # self.alignment_ref = np.array(
        #     [
        #         [-0.98694063, -0.02886373, 0.15847737],
        #         [0.07431554, 0.79127546, 0.60692698],
        #         [-0.14291743, 0.61077822, -0.77879687],
        #     ]
        # )
        # 30 deg endoscope
        # self.alignment_ref = np.array(
        #     [
        #         [-0.99984756, 0.01537409, -0.0082761],
        #         [0.01090215, 0.91996043, 0.39185961],
        #         [0.01363817, 0.39170965, -0.9199878],
        #     ]
        # )

        self.alignment_ref = np.array(
            [
                [-0.01846654, 0.9994546, -0.02737687],
                [0.87401319, 0.0294341, 0.48500988],
                [0.48555117, -0.0149713, -0.87408005],
            ]
        )

        x = X_actual.rotation[:, 0]
        y = X_actual.rotation[:, 1]

        print("x: ", x)
        print("y: ", y)

        self.alignment_ref

        p = np.array([0.42501, 0.69952, 0.328031])

        # Extract ux and uy from R
        self.ux = self.alignment_ref[:, 0]
        self.uy = self.alignment_ref[:, 1]
        # The z-axis of the rotation matrix R serves as the normal vector to the plane
        normal_vector = self.alignment_ref[:, 2]
        # Calculate the diagonal vectors
        # theta = self.getAlignmentAngle(self.ux, self.uy, x, y, normal_vector)
        px = self.project_onto_plane(x, normal_vector)
        py = self.project_onto_plane(y, normal_vector)
        print("px: ", px)
        print("py: ", py)
        theta = self.getIVPConstraints(px, py)

        if np.abs(theta) < 0.002:
            theta = 0.0
        return theta

    def create_pose_from_twist(self, current_pose, twist, rcm_pos):
        global roll_angle
        # get first 3 values from twist
        lin_vel = twist[0:3]
        point = np.array(current_pose.translation) + twist[0:3] * 0.001
        vz = point - rcm_pos
        uz = vz / np.linalg.norm(vz)

        uy = -np.cross(uz, [1, 0, 0])
        ux = np.cross(uy, uz)

        # Create rotation matrix with ux, uy, uz as columns
        path_ori = np.column_stack((ux, uy, uz))
        # print("ux: ", ux)
        # print("uy: ", uy)
        # print("uz: ", uz)

        r = R.from_matrix(path_ori)
        q = r.as_quat()
        q = q / np.linalg.norm(q)
        path_ori = R.from_quat(q).as_matrix()

        if twist[5] != 0 or roll_angle != 0:
            # if not self.saved:
            #     self.storeAxisProjections()
            roll_angle += twist[5] * np.pi / 180.0

            r2 = R.from_euler("z", roll_angle, degrees=False)

            # apply the rotation
            path_ori = path_ori @ r2.as_matrix()
            # Extract ux and uy (assuming they are the first and second columns)
            ux = path_ori[:, 0]
            uy = path_ori[:, 1]
        pose = pin.SE3(path_ori, point)

        return pose, ux, uy

    def sendJointGoalToActionServer(self, q_sol):
        client = actionlib.SimpleActionClient(
            "/unit0/FollowJointTarget", FollowJointTargetAction
        )
        client.wait_for_server()
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = self.model.names[1 : self.n_arm_joints + 1]
        msg.name.append("forceps1")
        msg.name.append("forceps2")
        msg.position = q_sol[: self.n_arm_joints]
        msg.position = np.append(msg.position, 0)
        msg.position = np.append(msg.position, 0)

        msg.velocity = np.zeros(self.n_arm_joints)
        msg.effort = np.zeros(self.n_arm_joints)
        goal = FollowJointTargetGoal()
        goal.joint_target = msg
        # Set your goal parameters here

        client.send_goal(goal)
        client.wait_for_result(rospy.Duration(0.1))
        # result = client.get_result()
        rospy.loginfo("goal sent to action server")

    def sendOperSpaceTrajGoalToActionServer(self, q_sol):
        client = actionlib.SimpleActionClient(
            "/unit0/FollowJointTarget", FollowJointTargetAction
        )
        client.wait_for_server()
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = self.model.names[1 : self.n_arm_joints + 1]
        msg.name.append("forceps1")
        msg.name.append("forceps2")
        msg.position = q_sol[: self.n_arm_joints]
        msg.position = np.append(msg.position, 0)
        msg.position = np.append(msg.position, 0)

        msg.velocity = np.zeros(self.n_arm_joints)
        msg.effort = np.zeros(self.n_arm_joints)
        goal = FollowJointTargetGoal()
        goal.joint_target = msg
        # Set your goal parameters here

        client.send_goal(goal)
        client.wait_for_result(0.01)
        # result = client.get_result()
        rospy.loginfo("goal sent to action server")

    def getGreenObjCb(self, msg):
        self.green_obj = np.array([msg.x, msg.y])
        self.object_detected = True
        # print("has green obj :", self.green_obj)

    def getTwistCb(self, msg):
        twist = np.array(
            [
                msg.linear.x,
                msg.linear.y,
                msg.linear.z,
                msg.angular.x,
                msg.angular.y,
                msg.angular.z,
            ]
        )

        pose, ux, uy = self.create_pose_from_twist(
            self.B_X_END, twist, self.rcm_position
        )
        if self.q_start is None:
            self.q_start = self.q0
        # call solver
        sol_found, q_sol, _, _, mp, summary = self.solveIK(self.q_start, pose)
        # publish if ik is solved.
        # print("ee pose :", self.B_X_END)
        rospy.logerr("----------------")
        print("input was :", pose)
        if sol_found:
            rospy.logwarn("IK solved!")
            # self.publish_qd(q_sol)
            # res.valid = True

            # if self.saved:
            rot = self.B_X_END.rotation
            ux = rot[:, 0]
            uy = rot[:, 1]
            theta = self.estimateMisallignment(ux, uy)
            self.publish_qd(q_sol)
            self.q_start = q_sol
            # if abs(theta) > 0:
            #     new_targ_pose = pose.copy()
            #     new_targ_pose.rotation = self.B_X_END.rotation * R.from_euler('z', theta, degrees=False).as_matrix()
            #     sol_found2, q_sol2, _, _, mp, summary = self.solveIK(self.q0, new_targ_pose)
            #     if sol_found2:
            #         rospy.logerr("solved with correction")
            #         pose = new_targ_pose
            #         self.publish_qd(q_sol2)
            #         # return
            #     else :
            #         rospy.logerr("not solved with correction")
            #         self.publish_qd(q_sol)

            #     # print("not solved with correction")
            #     # self.publish_qd(q_sol)
            #     rospy.loginfo("theta  > 0")
        else:
            rospy.logerr("IK not solved!")
            # print("Iniit not solved input was :", pose)
        rospy.logerr("----------------")
        print("joint values in degrees :", q_sol * 180 / np.pi)
        rospy.loginfo("----------------")
        print("joint values in degrees :", q_sol * 180 / np.pi + 360)
        rospy.logwarn("----------------")

    def getIKCb(self, req):
        # convert pose to SE3
        pose = pose_to_se3(req.checkPose)
        q_sol = self.q_endoscope
        # call solver
        sol_found, q_sol, _, _, mp, summary = self.solveIK(q_sol, pose)
        # publish if ik is solved.
        rospy.logerr("----------------")
        print("input was :", pose)
        res = ikCheckerResponse()
        if sol_found:
            rospy.logwarn("IK solved!")
            # res.valid = True
            self.publish_endoscope_qd(q_sol)
        else:
            rospy.logwarn("IK not solved!")
            print(" not solved input was :", pose)

        return res

    def track_SO3_trajectory_cubic_splines(
        self, q_init: np.ndarray, X_act: pin.SE3, X_des: pin.SE3
    ):
        kMaxLinVelEE = 0.005  # [mm/s]
        kMaxAngVelEE = 0.03  # [rad/s]
        kMaxCartAcc = 100.0  # [mm/s2]

        start_traj_time = time.time()

        # cycle_t_ = 0.1  # [s]

        B_Tw_EE_EEdes = pin.log6(X_des.act(X_act.inverse())).vector
        # print("Log6 of delta Tw: ", B_Tw_EE_EEdes)
        B_w_EE_EEdes = pin.log3(X_des.rotation @ np.transpose(X_act.rotation))
        # print("Log3 of delta w: ", np.linalg.norm(B_w_EE_EEdes))
        # print(
        #     "Norm of delta Tw: ", np.linalg.norm(X_des.translation - X_act.translation)
        # )

        # # Compute trajectory time
        # traj_time = N * cycle_t_
        TNv = np.linalg.norm(X_des.translation - X_act.translation) / (kMaxLinVelEE)
        TNw = np.linalg.norm(B_w_EE_EEdes) / (kMaxAngVelEE)

        # Select trajectory time as the maximum between the two
        traj_time = max(TNw, TNv)  # [s]

        print(
            "Traj.v (s): ",
            TNv,
            "Traj.w (s): ",
            TNw,
            "Total. Traj. Time (s): ",
            traj_time,
        )

        # Get elapsed trajectory time
        traj_actual_time = time.time() - start_traj_time
        # [s]

        s_traj = 3 * np.power(traj_actual_time / traj_time, 2) - 2 * np.power(
            traj_actual_time / traj_time, 3
        )

        print(
            "\n+++++++++++++\n Actual Traj. time (s): ",
            traj_actual_time,
            " traj_time:",
            traj_time,
            " s_traj [0-1]: ",
            s_traj,
            "\n++++++++++++++++++++\n",
        )

        q_sol = q_init

        while s_traj < 1:
            # Get elapsed trajectory time
            traj_actual_time = time.time() - start_traj_time
            # [s]

            print(
                "\n+++++++++++++\nTrajectory time: ",
                traj_actual_time,
                " s ++++++++++++++++++++\n",
            )

            s_traj = 3 * np.power(traj_actual_time / traj_time, 2) - 2 * np.power(
                traj_actual_time / traj_time, 3
            )
            if traj_actual_time > traj_time:
                s_traj = 1

            # Compute next SE3 step pose
            Xd_step = pin.exp6(B_Tw_EE_EEdes * s_traj) * X_act

            print("Xd_step: ", Xd_step)

            sol_found, q_sol, err_ee = self.solveIK_EE(q_sol, Xd_step)
            if sol_found:
                self.sendJointGoalToActionServer(q_sol)
                self.B_X_END = copy.deepcopy(self.data.oMf[self.ee_id])

                # self.publish_qd(q_sol)
                print("Ik EE solution found with EE error: ", err_ee)
            else:
                print("IK EE not solved")

    def solveIK_EE(self, q_init, B_X_des_Fee):
        print("-----------------------------\nSolving IK for EE")
        # # Target Pose
        # pos_target = np.array([0.41829602, -0.23521243, 0.21573797])
        # # ori_target = np.array(
        # #     [
        # #         [0.89829329, 0.31010489, -0.31129427],
        # #         [-0.22558082, 0.93344249, 0.27892368],
        # #         [0.3770709, -0.18033325, 0.90845884],
        # #     ]
        # # )

        # ori_target = np.array(
        #     [
        #         [0.0, -1.0, 0],
        #         [-1, 0.0, 0],
        #         [0.0, 0.0, -1],
        #     ]
        # )

        # B_X_des_Fee = pin.SE3(ori_target, pos_target)
        sol_found = False

        self.B_X_des_Fee = B_X_des_Fee
        q_sol = q_init

        # q_init = pin.randomConfiguration(model)
        # q_init = np.array([2.63, 0.87, -1.24, 2.03, -3.66, -1.05, -2.28, 2.63, 0.87])
        # q_init = self.q

        # Perform the forward kinematics over the kinematic tree
        pin.framesForwardKinematics(self.model, self.data, q_init)

        # Print initial configuration
        # print("Initial configuration: ", q_init)

        # Print EE pose
        # print("Initial EE position:", self.data.oMf[self.ee_id].translation)
        # print("Initial EE orientation:", self.data.oMf[self.ee_id].rotation)
        # print(
        #     "Is Initial EE orientation normalized: ",
        #     self.data.oMf[self.ee_id].rotation
        #     @ np.transpose(self.data.oMf[self.ee_id].rotation),
        # )
        # Print EE desired pose
        # print("Desired EE position:", self.B_X_des_Fee.translation)

        # self.B_X_des_Fee.rotation = self.normalize_rotation(self.B_X_des_Fee.rotation)
        # print("Desired EE orientation:", self.B_X_des_Fee.rotation)

        # print(
        #     "Is Desired EE orientation normalized: ",
        #     self.B_X_des_Fee.rotation @ np.transpose(self.B_X_des_Fee.rotation),
        # )

        # Initial guess
        q_it = q_init

        # For Task 2
        print("\n---------------\n Task EE \n------------------------")

        # Compute Task EE residual
        res_ee, err_ee = self.compute_residual_EE_log6()

        metric_performance = err_ee

        # Compute Task EE Jacobian
        B_Jb_END = self.compute_Jacobian_EE(q_it)

        # print("Initial EE residual", res_ee)
        # print("Initial EE error", err_ee)

        # Computing gradient
        # grad_mp = self.compute_GradManip_EE(q_it)
        # print("Initial Grad Manipulability index: ", grad_mp)

        start_ik = time.time()

        # For Priotity 1
        C_1 = np.zeros((2 * self.nq, self.nq))
        d_1 = np.zeros((2 * self.nq, 1))
        d_1_ext = np.zeros((3 * self.nq, 1))

        # For Priotity 2
        C_2 = np.zeros((2 * self.nq, self.nq))
        d_2 = np.zeros((2 * self.nq, 1))
        d_2_ext = np.zeros((3 * self.nq, 1))

        # For Priotity 3
        C_3 = np.zeros((2 * self.nq, self.nq))
        d_3 = np.zeros((2 * self.nq, 1))
        d_3_ext = np.zeros((3 * self.nq, 1))

        # Parameters stored in a single matrix for QP solver
        param = np.zeros((3 * self.nq, 5 * self.nq + 2))

        for it in range(200):
            # print("\n ---->  Iteration IK-EE solving", it)
            # print("--------- Solving Priority 1 --------------- ")
            start_test = time.time()

            # A_1 = np.transpose(B_Jb_Frcm.reshape(self.nq, 1))
            # b_1 = self.Kr["rcm"] * np.array([[res_rcm]])

            A_1 = copy.deepcopy(B_Jb_END)
            b_1 = self.Kr["ee"] * res_ee.reshape(6, 1)

            C_1[: self.nq, :] = np.eye(self.nq)
            C_1[self.nq :, :] = -np.eye(self.nq)

            x_max = np.minimum(self.q_max_ - q_it, 0.1 * np.ones((self.nq)))
            x_min = np.maximum(self.q_min_ - q_it, -0.1 * np.ones((self.nq)))

            # d_1[: self.nq, :] = (self.q_max_ - q_it).reshape(self.nq, 1)
            # d_1[self.nq :, :] = (-self.q_min_ + q_it).reshape(self.nq, 1)

            d_1[: self.nq, :] = x_max.reshape(self.nq, 1)
            d_1[self.nq :, :] = -x_min.reshape(self.nq, 1)

            Q_1, p_1, Cbar_1, dbar_1 = self.generate_QP_problem(
                1,
                1,
                [self.m_ee],
                np.eye(self.nq),
                q_it,
                [A_1],
                [b_1],
                C_1,
                d_1,
                [self.Kt["ee"]],
                self.Kd["p1_EE"],
                self.Kw["p1_EE"],
            )

            d_1_ext[: 2 * self.nq, :] = dbar_1

            param[:, : 3 * self.nq] = copy.deepcopy(Q_1)
            param[:, 3 * self.nq] = p_1.reshape(3 * self.nq)
            param[:, 3 * self.nq + 1 : 5 * self.nq + 1] = np.transpose(Cbar_1)
            param[:, param.shape[1] - 1]
            param = np.concatenate((Q_1, p_1, np.transpose(Cbar_1), d_1_ext), axis=1)

            r1 = self.qpsolver(
                p=param,
                lbg=0,
            )

            # print("--- %s seconds ---" % (time.time() - start_test))

            qd_opt_1 = r1["x"][0 : self.nq]
            # print("Optimal qd for Priority 1: ", qd_opt_1)

            # Verify if error is reducing
            # print("Jq: ", A_1 @ qd_opt_1)
            # print("Jq - e: ", A_1 @ qd_opt_1 - b_1)

            # Solution for Priority 1
            tw_opt_hat_1 = copy.deepcopy(qd_opt_1)

            # * Final solution
            # ? Only the first priority is considered
            qd_opt = copy.deepcopy(tw_opt_hat_1)

            # Integrating solution
            q_it = pin.integrate(self.model, np.array(q_it), np.array(qd_opt) * 1.0)
            # print("\nSolution (q):", q_it)

            # * Computing updated errors
            # FK
            pin.framesForwardKinematics(self.model, self.data, q_it)
            # print("New EE Pose:", self.data.oMf[self.ee_id])
            # print("Desired EE Pose:", self.B_X_des_Fee)

            # # Computing RCM residual
            # res_rcm, ps, pr, pe, err_rcm = self.compute_residual_RCM()

            # Computing EE residual
            res_ee, err_ee = self.compute_residual_EE_log6()

            # Computing Manipulability
            # mp = self.compute_Manip_EE(q_it)

            # print(
            #     "Iteration: ",
            #     it,
            #     # " RCM error: ",
            #     # err_rcm,
            #     " EE error: ",
            #     err_ee,
            #     # " Manipulability: ",
            #     # mp,
            # )

            # * Check for convergence
            # if err_rcm < self.eps_r and err_ee < self.eps_e:
            if err_ee < self.eps_e:
                print("Solution  found: ", q_it)
                q_sol = q_it
                sol_found = True
                break

            # Computing EE Jacobian
            B_Jb_END = self.compute_Jacobian_EE(q_it)

        print("TOTAL IK Solving--- %s seconds ---" % (time.time() - start_ik))

        return sol_found, q_sol, err_ee

    def solveIK_VT(self, q_init):
        # # Target Pose
        # pos_target = np.array([0.41829602, -0.23521243, 0.21573797])
        # # ori_target = np.array(
        # #     [
        # #         [0.89829329, 0.31010489, -0.31129427],
        # #         [-0.22558082, 0.93344249, 0.27892368],
        # #         [0.3770709, -0.18033325, 0.90845884],
        # #     ]
        # # )

        # ori_target = np.array(
        #     [
        #         [0.0, -1.0, 0],
        #         [-1, 0.0, 0],
        #         [0.0, 0.0, -1],
        #     ]
        # )

        # Camera sensor
        cam_pixel_size = 0.0000055
        cam_W = 2048
        cam_H = 1088

        # Camera intrinsic parameters
        fx = 1167.6213  # In pixels
        fy = 1176.00553  # In pixels
        cx = 1008  # In pixels
        cy = 536  # In pixels

        # ? For real camera
        f_lambda = (fx + fy) / 2  # In pixels
        self.s_center = np.array([cam_W / 2, cam_H / 2])
        # ? For simulation
        # f_lambda = self.compute_lambda(1080, 90)
        # self.s_center = np.array([540, 540])

        print("f_lambda: ", f_lambda)

        # Image feature target in pixels
        # self.s_target = np.array([1044, 800])

        # To center feature in image
        res_vt, err_vt = self.compute_residual_VT()
        print("res_vt: ", res_vt)

        # Parameters for the image Jacobian
        # ? Real camera
        # s_x = self.s_target[0] - cx
        # s_y = self.s_target[1] - cy
        # ? Simulation
        # s_x = self.green_obj[0]
        # s_y = self.green_obj[1]
        s_x = self.green_obj[1]
        s_y = -self.green_obj[0]

        # z = 0.18  # In meters
        z = 0.07  # In meters

        # Image Jacobian
        Jb_img = np.array(
            [
                [
                    -f_lambda / z,
                    0,
                    s_x / z,
                    s_x * s_y / f_lambda,
                    -(f_lambda**2 + s_x**2) / f_lambda,
                    s_y,
                ],
                [
                    0,
                    -f_lambda / z,
                    s_y / z,
                    (f_lambda**2 + s_y**2) / f_lambda,
                    -s_x * s_y / f_lambda,
                    -s_x,
                ],
            ]
        )

        # Jb_img = np.array([[-2447.10470, 0, 1014.20985],[0, -2422.18638 , 471.114582]])

        # B_X_des_Fee = pin.SE3(ori_target, pos_target)
        sol_found = False

        it_improvement = 0.0
        cnt_it_no_improvement = 0
        max_it_no_improvement = 50

        collision_mode = False
        # collision_mode = True

        metric_performance = 1e6
        dc = np.zeros((3, 1))

        # self.B_X_des_Fee = B_X_des_Fee
        q_sol = q_init

        # q_init = pin.randomConfiguration(model)
        # q_init = np.array([2.63, 0.87, -1.24, 2.03, -3.66, -1.05, -2.28, 2.63, 0.87])
        # q_init = self.q

        # Perform the forward kinematics over the kinematic tree
        pin.framesForwardKinematics(self.model, self.data, q_init)

        B_X_initial_END = copy.deepcopy(self.data.oMf[self.ee_id])

        # Print initial configuration
        print("Initial configuration: ", q_init)

        # Print EE pose
        print("Initial EE position:", self.data.oMf[self.ee_id].translation)
        print("Initial EE orientation:", self.data.oMf[self.ee_id].rotation)

        # Print EE desired pose
        # print("Desired EE position:", self.B_X_des_Fee.translation)
        # print("Desired EE orientation:", self.B_X_des_Fee.rotation)

        # Initial guess
        q_it = q_init

        # For Task 1
        print("\n---------------\n Task RCM \n------------------------")

        # Compute Task RCM residual
        res_rcm, ps, pr, pe, err_rcm, B_p_RCM = self.compute_residual_RCM()

        # Compute Task RCM Jacobian
        B_Jb_Frcm = self.compute_Jacobian_RCM(q_it, ps, pr, pe)

        print("Initial RCM residual", res_rcm)
        print("Initial RCM error", err_rcm)

        # For Task 2
        print("\n---------------\n Task VT \n------------------------")

        # Compute Task EE residual
        res_ee, err_ee = self.compute_residual_EE_log6()
        res_vt, err_vt = self.compute_residual_VT()

        metric_performance = err_ee

        # Compute Task VT Jacobian
        B_X_CAM = copy.deepcopy(self.data.oMf[self.pre_rcm_id])
        B_X_END = copy.deepcopy(self.data.oMf[self.ee_id])
        B_p_TROCAR = copy.deepcopy(self.B_p_Ftrocar)
        B_X_TIP = copy.deepcopy(self.data.oMf[self.post_rcm_id])

        END_p_TROCAR = B_X_END.inverse() * B_p_TROCAR
        END_p_RCM = B_X_END.inverse() * B_p_RCM

        # print("B_X_END: ", B_X_END)
        # print("B_p_TROCAR: ", B_p_TROCAR)
        # print("B_p_RCM: ", B_p_RCM)
        # print("B_X_CAM: ", B_X_CAM)
        # print("B_X_TIP: ", B_X_TIP)

        B_X_RCM = pin.SE3(self.B_R_TROCAR, B_p_TROCAR)
        # B_X_RCM = pin.SE3(B_X_TIP.rotation, B_p_TROCAR)
        # B_X_RCM = pin.SE3(B_X_TIP.rotation, B_p_RCM)

        END_X_RCM = B_X_END.inverse() * B_X_RCM
        # print("END_X_RCM: ", END_X_RCM)

        Ad_END_X_RCM = END_X_RCM.toActionMatrix()
        # print("Ad_END_X_RCM: ", Ad_END_X_RCM)

        # Create a 6x6 nupmy array where the upper left 3x3 is identiry and bottom left 3x3 is zero matrix
        # bottom right 3x3 is identity and upper right 3x3 is the skew symmetric matrix of the vector p
        END_Tw_hat_RCM_END = np.block(
            [
                # [np.eye(3), -pin.skew(END_p_TROCAR)],
                [np.eye(3), np.zeros((3, 3))],
                # [np.eye(3), pin.skew(END_p_RCM)],
                [np.zeros((3, 3)), np.eye(3)],
            ]
        )
        # print("END_Tw_hat_RCM_END: ", END_Tw_hat_RCM_END)

        # Jb_rcm = (END_Tw_hat_RCM_END @ Ad_END_X_RCM)[:, 2:]
        Jb_rcm = END_Tw_hat_RCM_END @ Ad_END_X_RCM
        # print("Jb_rcm: ", Jb_rcm)

        B_Jb_END = self.compute_Jacobian_EE(q_it)
        # # END_Adj_B = pin.SE3.Adj(self.data.oMf[self.ee_id].inverse())
        END_X_B = copy.deepcopy(self.data.oMf[self.ee_id].inverse())

        # # END_Adj_B = pin.SE3.Adj(END_X_B)
        END_Adj_B = END_X_B.toActionMatrix()
        # Jb_vt = Jb_img @ END_Adj_B @ B_Jb_END

        Jb_vt = Jb_img @ Jb_rcm

        # print("Initial EE residual", res_ee)
        # print("Initial EE error", err_ee)

        print("Initial VT residual", res_vt)
        print("Initial VT error", err_vt)

        # For Task 3
        print("\n---------------\n Task Misalignment \n------------------------")

        res_al = self.estimateMisallignment(self.data.oMf[self.ee_id])
        print("Initial Misalignment residual", res_al)

        # Jb_al = np.array([[0, 0, 0, 0, 0, 1]]) @ END_Adj_B @ B_Jb_END
        Jb_al = np.array([[0, 0, 0, 0, 0, -1]]) @ Jb_rcm

        print("\n---------------\n Task Depth \n------------------------")
        # can update depth here ...
        res_depth, err_depth, lhat = self.compute_residual_depth(0.09)
        print("Initial Depth error", err_depth)
        # Jb_depth = self.compute_Jacobian_depth(q_it, lhat)
        Jb_depth = np.array([[0, 0, -1, 0, 0, 0]]) @ Jb_rcm

        print("\n---------------\n Task Manip \n------------------------")
        # Compute Task Manip residual
        mp = self.compute_Manip_EE(q_it)
        print("Initial Manipulability index: ", mp)

        # Computing gradient
        grad_mp = self.compute_GradManip_EE(q_it)
        # print("Initial Grad Manipulability index: ", grad_mp)
        n_dof = 4

        start_ik = time.time()

        # For Priotity 1
        C_1 = np.zeros((2 * n_dof, n_dof))
        d_1 = np.zeros((2 * n_dof, 1))
        d_1_ext = np.zeros((3 * n_dof, 1))

        # For Priotity 2
        C_2 = np.zeros((2 * n_dof, n_dof))
        d_2 = np.zeros((2 * n_dof, 1))
        d_2_ext = np.zeros((3 * n_dof, 1))

        # For Priotity 3
        C_3 = np.zeros((2 * n_dof, n_dof))
        d_3 = np.zeros((2 * n_dof, 1))
        d_3_ext = np.zeros((3 * n_dof, 1))

        # Parameters stored in a single matrix for QP solver
        param = np.zeros((3 * n_dof, 5 * n_dof + 2))

        for it in range(1):
            print("\n ---->  Iteration ", it)
            print("--------- Solving Priority 1 --------------- ")
            start_test = time.time()

            # ? RCM
            # A_1 = np.transpose(B_Jb_Frcm.reshape(n_dof, 1))
            # b_1 = self.Kr["rcm"] * np.array([[res_rcm]])

            # ? End-effector pose tracking
            # A_1 = copy.deepcopy(B_Jb_END)
            # b_1 = self.Kr["ee"] * res_ee.reshape(6, 1)

            # ? Visual tracking
            A_1 = copy.deepcopy(Jb_vt[:, 2:])
            b_1 = Kr["vt"] * res_vt.reshape(2, 1)

            C_1[:n_dof, :] = np.eye(n_dof)
            C_1[n_dof:, :] = -np.eye(n_dof)

            # x_max = np.minimum(self.q_max_ - q_it, 0.1 * np.ones((n_dof)))
            # x_min = np.maximum(self.q_min_ - q_it, -0.1 * np.ones((n_dof)))

            # d_1[: n_dof, :] = (self.q_max_ - q_it).reshape(n_dof, 1)
            # d_1[n_dof :, :] = (-self.q_min_ + q_it).reshape(n_dof, 1)

            x_max = np.array([0.01, 0.1, 0.1, 0.1])
            x_min = np.array([-0.01, -0.1, -0.1, -0.1])

            d_1[:n_dof, :] = x_max.reshape(n_dof, 1)
            d_1[n_dof:, :] = -x_min.reshape(n_dof, 1)

            print("A_1: ", A_1)
            print("b_1: ", b_1)
            print("C_1: ", C_1)
            print("d_1: ", d_1)
            Q_1, p_1, Cbar_1, dbar_1 = self.generate_QP_problem_OperationSpace(
                1,
                1,
                [self.m_vt],
                np.eye(n_dof),
                np.zeros((n_dof, 1)),
                [A_1],
                [b_1],
                C_1,
                d_1,
                [self.Kt["vt"]],
                self.Kd["p1"],
                self.Kw["p1"],
            )

            d_1_ext[: 2 * n_dof, :] = dbar_1

            param[:, : 3 * n_dof] = copy.deepcopy(Q_1)
            param[:, 3 * n_dof] = p_1.reshape(3 * n_dof)
            param[:, 3 * n_dof + 1 : 5 * n_dof + 1] = np.transpose(Cbar_1)
            param[:, param.shape[1] - 1]
            param = np.concatenate((Q_1, p_1, np.transpose(Cbar_1), d_1_ext), axis=1)

            print(param)
            r1 = self.qpsolver_operational(
                p=param,
                lbg=0,
            )

            # print("--- %s seconds ---" % (time.time() - start_test))

            tw_opt_1 = r1["x"][0:n_dof]
            print("Optimal qd for Priority 1: ", tw_opt_1)

            # Verify if error is reducing
            # print("Jq: ", A_1 @ tw_opt_1)
            # print("Jq - e: ", A_1 @ tw_opt_1 - b_1)
            # print("Tw_opt_1: ", tw_opt_1)

            # Update estimated error
            err_vt = np.linalg.norm(A_1 @ tw_opt_1 - b_1)

            # Solution for Priority 1
            tw_opt_hat_1 = copy.deepcopy(tw_opt_1)

            # # * ---------------Priority 2--------------------------
            print("\--------- Solving Priority 2 --------------- ")

            # # # Computing Null Projector for Task 1
            N_1 = np.eye(n_dof) - np.linalg.pinv(A_1) @ A_1
            print(N_1)
            # ? Visual tracking
            # A_2 = copy.deepcopy(Jb_vt[:, 2:])
            # b_2 = Kr["vt"] * res_vt.reshape(2, 1)

            # ? End-effector pose tracking
            # A_2 = copy.deepcopy(B_Jb_END)
            # b_2 = self.Kr["ee"] * res_ee.reshape(6, 1)

            # # ? If Task 3 (manip) is added to Task 2 (EE) for Priority 2
            # A_3 = self.dT * np.transpose(grad_mp)
            # b_3 = self.Kr["manip"] * np.array([[mp]])

            # # ? For misalignment
            A_2 = copy.deepcopy(Jb_al[:, 2:])
            print("A_2: ", A_2)
            # b_2 = Kr["align"] * -self.estimateMisallignment(self.data.oMf[self.ee_id])
            b_2 = Kr["align"] * res_al
            print("b_2: ", b_2)

            C_2[:n_dof, :] = np.eye(n_dof)
            C_2[n_dof:, :] = -np.eye(n_dof)

            # x_max = np.minimum(self.q_max_ - q_it, 0.1 * np.ones((n_dof)))
            # x_min = np.maximum(self.q_min_ - q_it, -0.1 * np.ones((n_dof)))

            x_max = np.array([0.01, 0.1, 0.1, 0.1])
            x_min = np.array([-0.01, -0.1, -0.1, -0.1])
            # print("x_max: ", x_max)
            # print("x_min: ", x_min)

            d_2[:n_dof, :] = x_max.reshape(n_dof, 1)
            d_2[n_dof:, :] = -x_min.reshape(n_dof, 1)

            # Only One additional task
            Q_2, p_2, Cbar_2, dbar_2 = self.generate_QP_problem_OperationSpace(
                2,
                1,
                [self.m_align],
                N_1,
                tw_opt_hat_1,
                [A_2],
                [b_2],
                C_2,
                d_2,
                [self.Kt["align"]],
                self.Kd["p2"],
                self.Kw["p2"],
            )

            # # EE + Manip
            # # Q_2, p_2, Cbar_2, dbar_2 = self.generate_QP_problem(
            # #     2,
            # #     2,
            # #     [self.m_ee, self.m_manip],
            # #     N_1,
            # #     tw_opt_hat_1,
            # #     [A_2, A_3],
            # #     [b_2, b_3],
            # #     C_2,
            # #     d_2,
            # #     [self.Kt["ee"], self.Kt["manip"]],
            # #     self.Kd["p2"],
            # #     self.Kw["p2"],
            # # )

            d_2_ext[: 2 * n_dof, :] = dbar_2

            param = np.zeros((3 * n_dof, 5 * n_dof + 2))

            param[:, : 3 * n_dof] = copy.deepcopy(Q_2)
            param[:, 3 * n_dof] = np.array(p_2).reshape(3 * n_dof)
            param[:, 3 * n_dof + 1 : 5 * n_dof + 1] = np.transpose(Cbar_2)
            param[:, param.shape[1] - 1]
            param = np.concatenate((Q_2, p_2, np.transpose(Cbar_2), d_2_ext), axis=1)

            # start_test = time.time()

            r2 = self.qpsolver_operational(
                p=param,
                lbg=0,
            )

            # print("--- %s seconds ---" % (time.time() - start_test))

            # Solution for Priority 2 without affecting performance of Priority 1
            # print("r2[x]: ", r2["x"])
            tw_opt_2 = r2["x"][:n_dof]
            # tw_opt_2 = np.array([-0.01, 0.0, 0.0, 0.0])
            tw_opt_2 = np.array(tw_opt_2).reshape(n_dof, 1)
            print("Optimal qd for Priority 2: ", tw_opt_2)

            # Complete solution considering Priority 1 and 2
            tw_opt_hat_2 = N_1 @ tw_opt_2 + tw_opt_hat_1
            print("Complete solution qd for Priority 1 and 2: ", tw_opt_hat_2)

            # Verify if error is reducing
            # print(
            #     "P2 Jq - e ",
            #     np.transpose(A_2 @ tw_opt_hat_2 - b_2) @ (A_2 @ tw_opt_hat_2 - b_2),
            # )
            # print("A_2 @ tw_opt_2 - b_2: ", A_2 @ tw_opt_2 - b_2)

            # * ---------------Priority 3--------------------------
            print("\--------- Solving Priority 3 --------------- ")

            # # Computing Null Projector for Priority 2
            N_2 = N_1 @ (np.eye(n_dof) - np.linalg.pinv(A_2 @ N_1) @ A_2 @ N_1)

            # ? For manipulability
            # A_3 = self.dT * np.transpose(grad_mp)
            # b_3 = self.Kr["manip"] * np.array([[mp]])

            # ? For misalignment
            # A_3 = Jb_al
            # b_3 = Kr["align"] * -self.estimateMisallignment(self.data.oMf[self.ee_id])

            # ? For depth
            A_3 = copy.deepcopy(Jb_depth[:, 2:])
            print("A_3: ", A_3)
            b_3 = Kr["depth"] * res_depth.reshape(1, 1)
            print("b_3: ", b_3)

            C_3[:n_dof, :] = np.eye(n_dof)
            C_3[n_dof:, :] = -np.eye(n_dof)

            x_max = np.array([0.01, 0.1, 0.1, 0.1])
            x_min = np.array([-0.01, -0.1, -0.1, -0.1])
            # print("x_max: ", x_max)
            # print("x_min: ", x_min)

            d_3[:n_dof, :] = x_max.reshape(n_dof, 1)
            d_3[n_dof:, :] = -x_min.reshape(n_dof, 1)

            # d_3[:n_dof, :] = (self.q_max_ - q_it).reshape(n_dof, 1)
            # d_3[n_dof:, :] = (-self.q_min_ + q_it).reshape(n_dof, 1)

            Q_3, p_3, Cbar_3, dbar_3 = self.generate_QP_problem_OperationSpace(
                3,
                1,
                [self.m_depth],
                # [self.m_align],
                N_2,
                tw_opt_hat_2,
                [A_3],
                [b_3],
                C_3,
                d_3,
                [self.Kt["depth"]],
                # [self.Kt["align"]],
                self.Kd["p3"],
                self.Kw["p3"],
            )
            print("Q_3: ", Q_3)
            print("p_3: ", p_3)
            print("Cbar_3: ", Cbar_3)
            print("dbar_3: ", dbar_3)

            d_3_ext[: 2 * n_dof, :] = dbar_3

            param[:, : 3 * n_dof] = copy.deepcopy(Q_3)
            param[:, 3 * n_dof] = np.array(p_3).reshape(3 * n_dof)
            param[:, 3 * n_dof + 1 : 5 * n_dof + 1] = np.transpose(Cbar_3)
            param[:, param.shape[1] - 1]
            param = np.concatenate((Q_3, p_3, np.transpose(Cbar_3), d_3_ext), axis=1)

            print(param)
            # start_test = time.time()

            r3 = self.qpsolver_operational(
                p=param,
                lbg=0,
            )

            # print("--- %s seconds ---" % (time.time() - start_test))

            # Soulution for Priority 3 without affecting performance of Priority 1 and 2
            tw_opt_3 = r3["x"][:n_dof]
            print("Optimal qd for Priority 3: ", tw_opt_3)

            # Complete solution considering Priority 1, 2 and 3
            tw_opt_hat_3 = N_2 @ tw_opt_3 + tw_opt_hat_2
            print("Complete solution qd for Priority 1 and 2 and 3: ", tw_opt_hat_3)

            # print("Jq: ", A_3 @ tw_opt_3)
            # print("Jq - e: ", A_3 @ tw_opt_3 - b_3)

            # * Final solution
            RCM_Tw_B_RCM_opt = np.zeros((6, 1))
            # ? Only the first priority is considered
            # qd_opt = copy.deepcopy(tw_opt_hat_1)
            # ? Only the first and second priority are considered
            # qd_opt = copy.deepcopy(tw_opt_hat_2)
            # ? All priorities are considered
            qd_opt = copy.deepcopy(tw_opt_hat_3)
            # ? Only the second priority is considered
            # qd_opt = copy.deepcopy(tw_opt_2)
            # ? Only the third priority is considered
            # qd_opt = copy.deepcopy(tw_opt_3)

            RCM_Tw_B_RCM_opt[2:, :] = copy.deepcopy(qd_opt)
            END_Tw_B_END_opt = Jb_rcm @ RCM_Tw_B_RCM_opt

            # END_Tw_B_END_opt = Jb_rcm @ tw_opt_2

            B_Tw_B_END_opt = B_X_END.toActionMatrix() @ END_Tw_B_END_opt
            # print("B_Tw_B_END_opt: ", B_Tw_B_END_opt)
            # print("Norm(B_Tw_B_END_opt): ", np.linalg.norm(B_Tw_B_END_opt))

            X_EE_des = pin.exp6(B_Tw_B_END_opt) * self.data.oMf[self.ee_id]
            # print("X_EE_actual: ", self.data.oMf[self.ee_id])
            print("X_EE_des: ", X_EE_des)

            sol_found, q_sol_EE, err_ee = self.solveIK_EE(q_it, X_EE_des)

            if sol_found:
                print("\VT IK Solution (q):", q_sol_EE)
                q_it = q_sol_EE
                q_sol = q_it
                sol_found = True
            else:
                rospy.logerr("No solution found for EE")

            # Integrating solution
            # q_it = pin.integrate(self.model, np.array(q_it), np.array(qd_opt) * 1.0)
            # q_it = pin.integrate(self.model, np.array(q_it), np.array(q_sol_new) * 1.0)

            # * Computing updated errors
            # FK
            pin.framesForwardKinematics(self.model, self.data, q_it)
            print("New EE position:", self.data.oMf[self.ee_id].translation)

            # ld = self.data.oMf[self.ee_id].translation - self.data.oMf[self.pre_rcm_id].translation
            # ld_hat = ld/np.linalg.norm(ld)

            # la = np.linalg.norm(self.data.oMf[self.ee_id].translation - self.rcm_position)
            # p_EE_des = self.data.oMf[self.ee_id].translation + (0.075 -la) * ld_hat

            # X_EE_des = pin.SE3(self.data.oMf[self.ee_id].rotation, p_EE_des)

            # sol_found, q_sol_new, err_ee = self.solveIK_EE(q_it, X_EE_des)
            # print("\nNew Solution (q):", q_sol_new)

            # Computing RCM residual
            res_rcm, ps, pr, pe, err_rcm, _ = self.compute_residual_RCM()

            # Computing EE residual
            res_ee, err_ee = self.compute_residual_EE_log6()

            # Computing VT residual
            res_vt, err_vt = self.compute_residual_VT()

            # Computing Misalignment residual
            res_al = self.estimateMisallignment(self.data.oMf[self.ee_id])

            # Computing Manipulability
            mp = self.compute_Manip_EE(q_it)

            # Copmpute depth error
            res_depth, err_depth, lhat = self.compute_residual_depth(0.09)

            print(
                "Iteration: ",
                it,
                " RCM error: ",
                err_rcm,
                " VT error: ",
                err_vt,
                " Misalignment error: ",
                res_al,
                " Depth error: ",
                err_depth,
                " Manipulability: ",
                mp,
            )

            X_act = B_X_initial_END
            X_des = self.data.oMf[self.ee_id]

            # Generate and track SO3 trajectory
            self.track_SO3_trajectory_cubic_splines(q_init, X_act, X_des)

            # * Check for convergence
            # if err_rcm < self.eps_r and err_ee < self.eps_e:
            # if err_vt < 100:
            # if err_rcm < self.eps_r and err_vt < 100:
            # if err_rcm < self.eps_r:
            #     print("Solution  found: ", q_it)
            #     q_sol = q_it
            #     sol_found = True
            #     break

            # Computing RCM Jacobian
            # B_Jb_Frcm = self.compute_Jacobian_RCM(q_it, ps, pr, pe)

            # Computing VT Jacobian
            # B_Jb_END = self.compute_Jacobian_EE(q_it)
            # END_X_B = self.data.oMf[self.ee_id].inverse()
            # END_Adj_B = END_X_B.toActionMatrix()
            # Jb_vt = Jb_img @ END_Adj_B @ B_Jb_END

            # Computing Misalignment Jacobian
            # Jb_al = np.array([[0, 0, 0, 0, 0, 1]]) @ END_Adj_B @ B_Jb_END

            # Computing manipulability gradient
            # grad_mp = DM(self.compute_GradManip_EE(q_it))

        print("TOTAL IK Solving--- %s seconds ---" % (time.time() - start_ik))

        results_summary = {
            "q_sol": q_sol,
            "sol_found": sol_found,
            "err_rcm": err_rcm,
            "err_ee": err_ee,
            "mp": mp,
            "B_X_END": self.data.oMf[self.ee_id].translation,
        }

        return sol_found, q_sol, err_rcm, err_ee, mp, results_summary


def create_6DCircle_path(center, radius, n_points):
    path_points = []
    for i in range(n_points):
        path_pos = center + radius * np.array(
            [
                np.cos((i / n_points) * 2 * np.pi),
                np.sin((i / n_points) * 2 * np.pi),
                0,
            ]
        )
        path_ori = np.array(
            [
                [0, -1, 0],
                [-1, 0, 0],
                [0, 0, -1],
            ]
        )
        point = pin.SE3(path_ori, path_pos)
        path_points.append(point)
    return path_points


def create_pose_from_twist(self, current_pose, twist, rcm_pos):
    global roll_angle
    # get first 3 values from twist
    lin_vel = twist[0:3]
    point = numpy.array(current_pose.translation) + twist[0:3] * 0.001
    vz = point - rcm_pos
    uz = vz / np.linalg.norm(vz)

    uy = -np.cross(uz, [1, 0, 0])
    ux = np.cross(uy, uz)

    # Create rotation matrix with ux, uy, uz as columns
    path_ori = np.column_stack((ux, uy, uz))
    # print("ux: ", ux)
    # print("uy: ", uy)
    # print("uz: ", uz)

    r = R.from_matrix(path_ori)
    q = r.as_quat()
    q = q / np.linalg.norm(q)
    path_ori = R.from_quat(q).as_matrix()

    if twist[5] != 0:
        roll_angle += twist[5] * np.pi / 180.0

        r2 = R.from_euler("z", roll_angle, degrees=False)

        # apply the rotation
        path_ori = path_ori @ r2.as_matrix()
    pose = pin.SE3(path_ori, point)
    return pose


def create_4DCircle_path(center, radius, n_points, rcm_pos):
    path_points = []
    for i in range(n_points):
        path_pos = center + radius * np.array(
            [
                np.cos((i / n_points) * 2 * np.pi),
                np.sin((i / n_points) * 2 * np.pi),
                0,
            ]
        )

        vz = path_pos - rcm_pos
        uz = vz / np.linalg.norm(vz)

        # uy = np.cross(uz, [0, 1, 0])
        uy = -np.cross(uz, [1, 0, 0])
        ux = np.cross(uy, uz)

        # Create rotation matrix with ux, uy, uz as columns
        path_ori = np.column_stack((ux, uy, uz))
        # print("ux: ", ux)
        # print("uy: ", uy)
        # print("uz: ", uz)

        r = R.from_matrix(path_ori)
        q = r.as_quat()
        q = q / np.linalg.norm(q)
        path_ori = R.from_quat(q).as_matrix()

        point = pin.SE3(path_ori, path_pos)
        path_points.append(point)
    return path_points


if __name__ == "__main__":
    # Initialize ROS node
    rospy.init_node("hqp_collision", anonymous=True)

    # Get the robot id
    robot_id = rospy.get_param("~robot_id", "asar_endoscope")

    # Get the urdf path
    urdf_path = get_urdf_path(robot_id)

    # Get the integration time
    dT = rospy.get_param("~dT", 0.0001)

    # Get the end-effector link name
    ee_link_name = rospy.get_param("~ee_link_name", "endoscope_link")

    # Get RCM pre and post joint names
    pre_rcm_joint_name = rospy.get_param("~pre_rcm_joint_name", "joint_camera")
    post_rcm_joint_name = rospy.get_param("~post_rcm_joint_name", "joint_tip")

    pre_link1_joint_name = rospy.get_param("~pre_link1_joint_name", "joint_interface")
    post_link1_joint_name = rospy.get_param("~post_link1_joint_name", "rst_joint_1")

    pre_link2_joint_name = rospy.get_param("~pre_link2_joint_name", "rst_joint_2")
    post_link2_joint_name = rospy.get_param("~post_link2_joint_name", "joint_ee")

    # Get the error tolerances
    eps_r = rospy.get_param("~eps_r", 0.0001)
    eps_e = rospy.get_param("~eps_e", 0.00001)
    eps_v = rospy.get_param("~eps_v", 0.001)
    eps_w = rospy.get_param("~eps_w", 0.001)

    # Get the RCM position
    rcm_pos = rospy.get_param("~rcm_pos", [0.440, 0.365, 0.264])

    # Get number of joints in the arm
    n_arm_joints = rospy.get_param("~n_arm_joints", 7)

    # Get number of joints in the RST
    n_rst_joints = rospy.get_param("~n_rst_joints", 0)

    # Get if manipulability is used
    is_manip_on = rospy.get_param("~is_manip_on", False)

    # Get Task coefficients Kt
    Kt_rcm = rospy.get_param("~Kt_rcm", 1.0)  #
    Kt_ee = rospy.get_param("~Kt_ee", 1.0)  #
    Kt_coll = rospy.get_param("~Kt_coll", 1.0)  #
    Kt_manip = rospy.get_param("~Kt_manip", 1)  #
    Kt_vt = rospy.get_param("~Kt_vt", 1)  #
    Kt_align = rospy.get_param("~Kt_align", 1)  #
    Kt_depth = rospy.get_param("~Kt_depth", 1)  #

    Kt = dict()
    Kt["rcm"] = Kt_rcm
    Kt["ee"] = Kt_ee
    Kt["coll"] = Kt_coll
    Kt["manip"] = Kt_manip
    Kt["vt"] = Kt_vt
    Kt["align"] = Kt_align
    Kt["depth"] = Kt_depth

    # Get residuals coefficients Kr
    Kr_rcm = rospy.get_param("~Kr_rcm", 1.0)  #
    Kr_ee = rospy.get_param("~Kr_ee", 1.0)  #
    Kr_coll = rospy.get_param("~Kr_coll", 1.0)  #
    Kr_manip = rospy.get_param("~Kr_manip", 0.01)  #
    Kr_vt = rospy.get_param("~Kr_vt", 0.05)  #
    Kr_align = rospy.get_param("~Kr_align", 0.1)  #
    Kr_depth = rospy.get_param("~Kr_depth", 0.1)  #

    Kr = dict()
    Kr["rcm"] = Kr_rcm
    Kr["ee"] = Kr_ee
    Kr["coll"] = Kr_coll
    Kr["manip"] = Kr_manip
    Kr["vt"] = Kr_vt
    Kr["align"] = Kr_align
    Kr["depth"] = Kr_depth

    # Get Joint-distance coefficients Kd
    Kd_p1 = rospy.get_param("~Kd_p1", 0.01)
    Kd_p2 = rospy.get_param("~Kd_p2", 0.1)
    Kd_p3 = rospy.get_param("~Kd_p3", 0.1)
    Kd_p4 = rospy.get_param("~Kd_p4", 0.00001)
    Kd_p1_EE = rospy.get_param("~Kd_p1_EE", 0.1)

    Kd = dict()
    Kd["p1"] = Kd_p1
    Kd["p2"] = Kd_p2
    Kd["p3"] = Kd_p3
    Kd["p4"] = Kd_p4
    Kd["p1_EE"] = Kd_p1_EE

    # Get slack variable coefficients Kw
    Kw_p1 = rospy.get_param("~Kw_p1", 10)
    Kw_p2 = rospy.get_param("~Kw_p2", 10)
    Kw_p3 = rospy.get_param("~Kw_p3", 10)
    Kw_p4 = rospy.get_param("~Kw_p4", 0.00001)
    Kw_p1_EE = rospy.get_param("~Kw_p1_EE", 10)

    Kw = dict()
    Kw["p1"] = Kw_p1
    Kw["p2"] = Kw_p2
    Kw["p3"] = Kw_p3
    Kw["p4"] = Kw_p4
    Kw["p1_EE"] = Kw_p1_EE

    # Create params dictionary
    params = {
        "dT": dT,
        "ee_link_name": ee_link_name,
        "pre_rcm_joint_name": pre_rcm_joint_name,
        "post_rcm_joint_name": post_rcm_joint_name,
        "pre_link1_joint_name": pre_link1_joint_name,
        "post_link1_joint_name": post_link1_joint_name,
        "pre_link2_joint_name": pre_link2_joint_name,
        "post_link2_joint_name": post_link2_joint_name,
        "eps.e": eps_e,
        "eps.v": eps_v,
        "eps.w": eps_w,
        "eps.r": eps_r,
        "Kt": Kt,
        "Kr": Kr,
        "Kd": Kd,
        "Kw": Kw,
        "rcm_pos": rcm_pos,
        "n_arm_joints": n_arm_joints,
        "n_rst_joints": n_rst_joints,
        "is_manip_on": is_manip_on,
    }

    # Instantiate the solver
    solver = HqpEndoscopeController(urdf_path, params)

    # Create the path
    center = np.array([0.425, 0.500, 0.328])
    radius = 0.04
    n_points = 10

    # center = np.array([0.433, -0.203, 0.211])
    # radius = 0.01
    # n_points = 400

    path_points = create_4DCircle_path(center, radius, n_points, rcm_pos)

    i = 0
    init_target_set = False
    target = pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.0]))

    controller_ready = False

    # Spin
    time.sleep(1)
    while not rospy.is_shutdown():
        if solver.is_arm0_q_init and solver.is_rst0_q_init and not controller_ready:
            solver.B_X_des_Fee = solver.B_X_END
            solver.B_R_TROCAR = solver.data.oMf[solver.post_rcm_id].rotation
            controller_ready = True
            print("Controller ready")

        if controller_ready:
            q_sol = solver.q0
            cnt = 0
            mp_record = []

            i += 1
            # if i > n_points - 1:
            #     break

            # for point in path_points:
            if solver.object_detected:
                # sol_found, q_sol, _, _, mp, summary = solver.solveIK(q_sol, point)
                sol_found, q_sol, _, _, mp, summary = solver.solveIK_VT(q_sol)
                if sol_found:
                    cnt += 1
                    print("[Solution found]: ", q_sol)

                    # time.sleep(1)
                    # solver.sendJointGoalToActionServer(q_sol)
                    # solver.publish_qd(q_sol)
                    mp_record.append(mp)

                    # # Save results in CSV file
                    # with open(csv_filename, "a") as f:
                    #     writer = csv.writer(f)
                    #     writer.writerow(
                    #         [
                    #             summary["sol_found"],
                    #             summary["err_rcm"],
                    #             summary["err_ee"],
                    #             summary["mp"],
                    #             summary["B_X_END"][0],
                    #             summary["B_X_END"][1],
                    #             summary["B_X_END"][2],
                    #             summary["q_sol"][0],
                    #             summary["q_sol"][1],
                    #             summary["q_sol"][2],
                    #             summary["q_sol"][3],
                    #             summary["q_sol"][4],
                    #             summary["q_sol"][5],
                    #             summary["q_sol"][6],
                    #         ]
                    #     )

                else:
                    rospy.logerr("Solution not found")
                #     exit()
                # time.sleep(0.5)

                break

    print("Solutions found: ", cnt)
    print("Avg. Manipulability: ", np.mean(mp_record))
    # solver.storeAxisProjections()
    # rospy.spin()
# %%


# %%
