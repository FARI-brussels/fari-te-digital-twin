# %%
import time
import numpy as np
import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf
import zmq
import struct

context = zmq.Context()
subscriber = context.socket(zmq.SUB)
# Connect to multiple publishers with different IPs
subscriber.connect("tcp://127.0.0.1:5555")
subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
# Subscriber
def recv_robot_data_fast(subscriber, num_joints=6):
    message = subscriber.recv()
    print(message)
    robot_id = struct.unpack('I', message[:4])[0]
    joint_bytes = message[4:]
    joint_positions = np.frombuffer(joint_bytes, dtype=np.float32)
    print("received message", robot_id, joint_positions)
    return robot_id, joint_positions

def main():
    """Main function for basic IK."""

    urdf = load_robot_description("panda_description")

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)

    # Set up visualizer.
    server = viser.ViserServer()
    
    server.scene.add_grid("/ground", width=2, height=2)
    with open("stand.glb", "rb") as f:
        glb_data = f.read()
    server.scene.add_glb("/stand_1", glb_data)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    while True:
        print("yooo")
        robot_id, joint_positions = recv_robot_data_fast(subscriber)
        
        urdf_vis.update_cfg(joint_positions)

if __name__ == "__main__":
    main()
# %%
