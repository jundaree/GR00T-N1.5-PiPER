import time
import numpy as np
import mujoco
import os
import shutil
from scipy.spatial.transform import Rotation as R
import argparse
import zmq
import msgpack
import io
from typing import Dict, Any

'''
Usage Examples

python inference_mujoco.py -l "pick the cube and place it in the red bowl"
python inference_mujoco.py -l "push the red bowl to the target location between two yellow lines"

'''

def render_rgb_depth(renderer, data, camera: str = "top_cam"):
    """Return (rgb uint8 HxWx3, depth float32 HxW in meters) from a named camera."""
    # update the scene from current data and camera
    renderer.update_scene(data, camera=camera)
    
    # RGB 
    renderer.disable_depth_rendering()
    renderer.disable_segmentation_rendering()
    rgb = renderer.render().copy()          

    renderer.enable_segmentation_rendering()

    # Depth
    renderer.enable_depth_rendering()
    renderer.disable_segmentation_rendering()
    depth = renderer.render().copy()        

    return rgb.copy(), depth.copy()

# Copy necessary classes from service.py
class MsgSerializer:
    @staticmethod
    def to_bytes(data: dict) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        if "__ndarray_class__" in obj:
            obj = np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


class ExternalRobotInferenceClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str = None,
    ):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> dict:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        try:
            self.socket.send(MsgSerializer.to_bytes(request))
            message = self.socket.recv()
            response = MsgSerializer.from_bytes(message)

            if "error" in response:
                raise RuntimeError(f"Server error: {response['error']}")
            return response
        except zmq.error.Again:
            print(f"Timeout waiting for response from server")
            self._init_socket()  # Recreate socket for next attempt
            raise
        except zmq.error.ZMQError as e:
            print(f"ZMQ Error: {e}")
            self._init_socket()  # Recreate socket for next attempt
            raise

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the action from the server.
        The exact definition of the observations is defined
        by the policy, which contains the modalities configuration.
        """
        return self.call_endpoint("get_action", observations)

    def __del__(self):
        """Cleanup resources on destruction"""
        self.socket.close()
        self.context.term()


class Gr00tRobotInferenceClient:
    def __init__(
        self,
        host="localhost",
        port=5555,
        language_instruction,
    ):
        self.language_instruction = language_instruction
        self.policy = ExternalRobotInferenceClient(host=host, port=port)

    def get_action(self, top_cam_rgb, side_cam_rgb, single_arm, gripper_position):
        """
        Query the GR00T inference server for actions.
        
        Args:
            top_cam_rgb: RGB image array (H, W, 3)
            side_cam_rgb: RGB image array (H, W, 3)
            single_arm: Joint positions array (6,)
            gripper_position: Gripper position (1,)
        
        Returns:
            dict: Action dictionary with keys like "action.single_arm", "action.gripper"
        """
        obs_dict = {
            "video.top_cam_rgb": top_cam_rgb[np.newaxis, :, :, :],  # (1, H, W, 3)
            "video.side_cam_rgb": side_cam_rgb[np.newaxis, :, :, :],  # (1, H, W, 3)
            "state.single_arm": single_arm[np.newaxis, :].astype(np.float64),  # (1, 6)
            "state.gripper": gripper_position[np.newaxis, :].astype(np.float64),  # (1, 1)
            "annotation.human.action.task_description": [self.language_instruction],
        }
        
        try:
            action_dict = self.policy.get_action(obs_dict)
            return action_dict
        except Exception as e:
            print(f"Failed to get action from server: {e}")
            return None

    def set_lang_instruction(self, lang_instruction):
        self.language_instruction = lang_instruction


def main(language_instruction: str):
    # --- Load model/data ---
    model = mujoco.MjModel.from_xml_path("./scene_pick_and_place.xml")  # Load the complete scene with robot and breadboard
    data  = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=480, width=640)

    # (Optional) viewer - Updated for MuJoCo 3.3.5
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(model, data)
    # ---- offscreen renderer (pick your size) ----

    # --- Helpers ---
    def actuator_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def joint_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

    # Build a stable name->index mapping for actuators
    # Updated for new robot: 6 arm joints + 1 gripper actuator
    act_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
    act_idx   = np.array([actuator_id(n) for n in act_names], dtype=int)

    def set_qpos_by_joint_names(qpos_targets: dict):
        """
        Initialize pose by directly setting joint positions, then forward the model.
        Use joint names (not actuator names) for clarity.
        """
        for name, q in qpos_targets.items():
            jid = joint_id(name)
            dof = model.jnt_qposadr[jid]
            data.qpos[dof] = q
        mujoco.mj_forward(model, data)

    # Extract actuator ctrl ranges for clamping (shape: [nu, 2])
    ctrl_range = model.actuator_ctrlrange.copy()

    # Print actuator information for debugging
    print("Available actuators:")
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  {i}: {actuator_name}")

    print(f"\nControl ranges:")
    for name in act_names:
        try:
            idx = actuator_id(name)
            lo, hi = ctrl_range[idx]
            print(f"  {name}: [{lo:.3f}, {hi:.3f}]")
        except:
            print(f"  {name}: NOT FOUND")

    def set_targets_by_dict(targets: dict):
        """
        targets: dict like {"joint1": 0.0, "joint2": 1.0, ...}
        Writes into data.ctrl in actuator order, with safe clamping.
        """
        for name, value in targets.items():
            i = actuator_id(name)
            lo, hi = ctrl_range[i]
            data.ctrl[i] = np.clip(value, lo, hi)
            
    # --- Example: home, move, and operate gripper ---
    # 1) Set a comfortable start pose (radians for revolute joints)
    # Updated for new robot: 6 arm joints + gripper joints
    home = {
        "joint1": 0.0,     # Base rotation
        "joint2": 0.0,     # Shoulder
        "joint3": 0.0,     # Elbow  
        "joint4": 0.0,     # Wrist roll
        "joint5": 0.0,     # Wrist pitch
        "joint6": 0.0,     # Wrist yaw
        "gripper": 0.0,     # Gripper (0 = closed, 0.035 = open)
    }
    # Set initial joint positions
    set_qpos_by_joint_names(home)

    # set the init pose of the bowl
    # Get index of the first qpos entry for this free joint
    bowl_qpos_id = model.joint(name="bowl_joint")
    bowl_qpos_addr = bowl_qpos_id.qposadr[0] # start index in data.qpos

    # Set new orientation
    euler_new = np.array([0., 0., np.pi/2])  # comment this line to randomize the board angle
    quat_new = R.from_euler('xyz', euler_new).as_quat()  # [x, y, z, w] order

    desired_pos = [0.5, 0.0, 0.04175]

    # Update qpos: [x, y, z, qw, qx, qy, qz]
    data.qpos[bowl_qpos_addr:bowl_qpos_addr+3] = desired_pos
    data.qpos[bowl_qpos_addr+3:bowl_qpos_addr+7] = [quat_new[3], quat_new[0], quat_new[1], quat_new[2]]

    # for cube
    cube_qpos_id = model.joint(name="cube_joint")
    cube_qpos_addr = cube_qpos_id.qposadr[0] # start index in data.qpos

    # Set new orientation
    rot_values = np.random.uniform(-0.05, 0.05, size=1)
    euler_new = np.array([0., 0., np.pi/2 + rot_values[0]])  # comment this line to randomize the board angle
    quat_new = R.from_euler('xyz', euler_new).as_quat()  # [x, y, z, w] order

    values = np.random.uniform(-0.05, 0.05, size=2)
    position_new = [0.35 + values[0], -0.20 + values[1], 0.030]



    # Update qpos: [x, y, z, qw, qx, qy, qz]
    data.qpos[cube_qpos_addr:cube_qpos_addr+3] = position_new
    data.qpos[cube_qpos_addr+3:cube_qpos_addr+7] = [quat_new[3], quat_new[0], quat_new[1], quat_new[2]]

    mujoco.mj_forward(model, data)

    # zero-out the simulator
    data.ctrl[:] = 0.0
    for t in range(200):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()

        time.sleep(0.001)  # Small delay for smooth visualization

    # Initialize GR00T client with ZMQ
    client = Gr00tRobotInferenceClient(
        host="localhost",
        port=5555,
        language_instruction=language_instruction
    )

    print(f"GR00T inference client initialized (ZMQ) with instruction: '{language_instruction}'")

    print("Begin simulation evaluation...")
    maximum_simulation_steps = 10000

    for t in range(maximum_simulation_steps):
        if viewer is not None:
            viewer.sync()

        rgb_top_cam, depth_img = render_rgb_depth(renderer, data, "top_cam")
        d = depth_img.copy()
        m = np.isfinite(d)
        if m.any():
            # normalize depth to [0, 255]
            d_vis = (255 * (d[m] - d[m].min()) / (np.ptp(d[m]) + 1e-8)).astype(np.uint8)
            depth_top_cam = np.zeros_like(d, dtype=np.uint8)
            depth_top_cam[m] = d_vis
        else:
            depth_top_cam = np.zeros_like(d, dtype=np.uint8)

        # rgb_top_cam, and depth_top_cam are the rbgd images of the top camera.
        rgb_side_cam, _ = render_rgb_depth(renderer, data, "side_cam")
        # rgb_side_cam is the rbgd image of the side camera.
        
        cur_robot_state = data.ctrl.copy()
        # cur_robot_state is the current robot state, which might also be the input to your trained model as long as you are not doing visual servoing.
        
        
        # Extract joint positions (first 6) and gripper (last 1)
        single_arm = cur_robot_state[:6]
        gripper_position = cur_robot_state[6:7]
        
        # Query the GR00T inference server
        action_dict = client.get_action(rgb_top_cam, rgb_side_cam, single_arm, gripper_position)
        
        if action_dict is not None:
            # Extract actions from response - action chunk has 16 timesteps
            if "action.single_arm" in action_dict and "action.gripper" in action_dict:
                # Execute all 16 actions in the chunk
                action_chunk_size = 8
                MODALITY_KEYS = ["single_arm", "gripper"]
                
                for action_idx in range(action_chunk_size):
                    # Concatenate actions using np.atleast_1d for proper dimension handling
                    actions = np.concatenate(
                        [np.atleast_1d(action_dict[f"action.{key}"][action_idx]) for key in MODALITY_KEYS],
                        axis=0,
                    )
                    
                    policy_command = {
                        "joint1": actions[0],     # Base rotation
                        "joint2": actions[1],     # Shoulder
                        "joint3": actions[2],     # Elbow  
                        "joint4": actions[3],     # Wrist roll
                        "joint5": actions[4],     # Wrist pitch
                        "joint6": actions[5],     # Wrist yaw
                        "gripper": actions[6],     # Gripper (0 = closed, 0.035 = open)
                    }
                    
                    set_targets_by_dict(policy_command)
                    mujoco.mj_step(model, data)
                    
                    if viewer is not None:
                        viewer.sync()
                    
                    # 30ms interval between actions
                    time.sleep(0.03)
                    
                # Check cube position after executing the action chunk
                current_cube_posi = data.qpos[cube_qpos_addr:cube_qpos_addr+3]
                print(f"Step {t}: Current cube pose: {current_cube_posi}")
                
                # Check if task is complete - cube being dropped off in the bowl
                if np.linalg.norm(desired_pos - current_cube_posi) <= 0.05:
                    print(f"Success! Cube placed in the bowl at step {t}")
                    break
            else:
                print(f"Warning: Unexpected action format from server at step {t}")
                # Fall back to current state - no action
                time.sleep(0.03)
        else:
            print(f"Warning: No action received from server at step {t}")
            # Fall back to current state - no action
            time.sleep(0.03)

    print("Finish simulation evaluation...")

    # Keep the viewer open
    if viewer is not None:
        print("Simulation complete. Close the viewer window to exit.")
        try:
            while viewer.is_running():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Simulation interrupted.")
        finally:
            viewer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MuJoCo simulation with GR00T inference')
    parser.add_argument(
        '-l', '--language_instruction',
        type=str,
        required=True,
        help='Language instruction for the robot task'
    )
    args = parser.parse_args()
    
    main(args.language_instruction)
