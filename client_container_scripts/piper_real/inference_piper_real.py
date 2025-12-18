import time
import numpy as np
import mujoco
import os
import shutil
from utils_solution import * 
from scipy.spatial.transform import Rotation as R
from piper_sdk import *
import msgpack
import io
from typing import Dict, Any
import argparse

'''
Usage Examples

python inference_piper_real.py -l "pick the cube and place it in the red bowl"
python inference_piper_real.py -l "push the red bowl to the target location between two yellow lines"

'''


# Add GR00T inference client classes
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
        """Call an endpoint on the server."""
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
            self._init_socket()
            raise
        except zmq.error.ZMQError as e:
            print(f"ZMQ Error: {e}")
            self._init_socket()
            raise

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Get the action from the server."""
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
        language_instruction="push the red bowl to the target location between two yellow lines",
    ):
        self.language_instruction = language_instruction
        self.policy = ExternalRobotInferenceClient(host=host, port=port)

    def get_action(self, top_cam_rgb, side_cam_rgb, single_arm, gripper_position):
        """Query the GR00T inference server for actions."""
        obs_dict = {
            "video.top_cam_rgb": top_cam_rgb[np.newaxis, :, :, :],
            "video.side_cam_rgb": side_cam_rgb[np.newaxis, :, :, :],
            "state.single_arm": single_arm[np.newaxis, :].astype(np.float64),
            "state.gripper": gripper_position[np.newaxis, :].astype(np.float64),
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


# --- Hardware Initialization ---
piper = C_PiperInterface_V2()
piper.ConnectPort()
while( not piper.EnablePiper()):
    time.sleep(0.01)
piper.MotionCtrl_2(0x01, 0x00, 7, 0x00)
piper.GripperCtrl(0,1000,0x01, 0)
factor = 57295.7795 #1000*180/3.1415926

# Real camera setups
import cv2 as cv
import pyrealsense2 as rs
import threading
import queue

def read_from_camera(cap, frame_queue, running):
    """ Meant to be run from a thread.  Loads frames into a global queue.

    Args:
        cap: OpenCV capture object (e.g., webcam)
        frame_queue: queue in which frames are put
        running: list containing a Boolean that determines if this function continues running
    """

    while running[0]:
        result = cap.read()
        frame_queue.put(result)

    print('read_from_camera thread stopped')


def get_robot_state():
    """Get current robot joint positions from Piper."""
    joint_state = piper.GetArmHighSpdInfoMsgs()
    gripper_state = piper.GetArmGripperMsgs().gripper_state.grippers_angle/1000000

    single_arm = np.zeros(6)
    for i in range(6):
        val = getattr(joint_state, f"motor_{i+1}").pos
        single_arm[i] = val / 1000.0  # Convert to radians
    return single_arm, np.array([gripper_state])


def main(language_instruction: str):
    realsense = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    realsense.start(config)

    logitech = cv.VideoCapture(6)
    if not logitech.isOpened():
        print("Could not open Logitech camera.  Exiting")
        return
    
    logitech.set(3, 640)
    logitech.set(4, 480)

    codec = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
    logitech.set(cv.CAP_PROP_FOURCC, codec)

    logitech.set(cv.CAP_PROP_FPS, 30)

    read_from_camera_running = [True]
    frame_queue = queue.Queue()
    read_from_camera_thread = threading.Thread(target=read_from_camera, args=(logitech, frame_queue, read_from_camera_running))
    read_from_camera_thread.start()

    joint_positions = [0,0,0,0,0,0,0]
    
    print("Initializing GR00T inference client...")
    client = Gr00tRobotInferenceClient(
        host="localhost",
        port=5555,
        language_instruction=language_instruction
    )
    print(f"GR00T inference client initialized with instruction: '{language_instruction}'")
    
    step_count = 0
    maximum_steps = 1000
    
    piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
    while(True):
        ret, frame = frame_queue.get(timeout=5)
        
        # ret, frame are now set and the queue is empty after this block
        while True:
            try:
                ret, frame = frame_queue.get_nowait()
            except queue.Empty:
                break

        if(not ret):
            print('Could not get frame')
            # Skip this iteration if there is not a valid frame
            continue

        frames = realsense.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue
        
        rs_rgb = np.asanyarray(color_frame.get_data())
        rs_depth = np.asanyarray(depth_frame.get_data())
        logi_rgb = frame

        
        
        # Get current robot state
        single_arm, gripper_position = get_robot_state()
        
        print(f"Step {step_count}: Current joint positions: {single_arm}")
        
        # Query the GR00T inference server
        action_dict = client.get_action(rs_rgb, logi_rgb, single_arm, gripper_position)
        
        if action_dict is not None:
            if "action.single_arm" in action_dict and "action.gripper" in action_dict:
                # Execute all 16 actions in the chunk
                action_chunk_size = 8
                for action_idx in range(action_chunk_size):
                    actions_arm = action_dict["action.single_arm"][action_idx]  # (6,)
                    actions_gripper = action_dict["action.gripper"][action_idx]  # (1,)
                    
                    # Update joint positions from action
                    for i in range(6):
                        joint_positions[i] = actions_arm[i]
                    joint_positions[6] = actions_gripper[0]
                    
                    # Move joints
                    joint_0 = round(joint_positions[0]*factor)
                    joint_1 = round(joint_positions[1]*factor)
                    joint_2 = round(joint_positions[2]*factor)
                    joint_3 = round(joint_positions[3]*factor)
                    joint_4 = round(joint_positions[4]*factor)
                    joint_5 = round(joint_positions[5]*factor)
                    joint_6 = round(joint_positions[6]*1000*1000)
                    piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
                    piper.GripperCtrl(joint_6, 1000, 0x01, 0)
                    
                    # 30ms interval between actions
                    time.sleep(0.05)
                
                print(f"Executed action chunk at step {step_count}")
            else:
                print(f"Warning: Unexpected action format from server at step {step_count}")
        else:
            print(f"Warning: No action received from server at step {step_count}")
        
        step_count += 1
        
        # Check if maximum steps reached
        if step_count >= maximum_steps:
            print("Maximum steps reached")
            break



        # Update video streams
        depth_colormap = cv.applyColorMap(cv.convertScaleAbs(rs_depth, alpha=0.03), cv.COLORMAP_JET)
        cv.imshow('rs_rgb', rs_rgb)
        cv.imshow('rs_depth', depth_colormap)
        cv.imshow('logi_rgb', logi_rgb)

        key = cv.waitKey(1)

        # If 'q' is pressed, quit the main loop
        if(key == ord('q')):
            break

    # Terminate the thread that's running in the background
    read_from_camera_running[0] = False
    read_from_camera_thread.join()

    logitech.release()
    realsense.stop()
    cv.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Piper robot hardware inference with GR00T')
    parser.add_argument(
        '-l', '--language_instruction',
        type=str,
        required=True,
        help='Language instruction for the robot task'
    )
    args = parser.parse_args()
    
    main(args.language_instruction)
