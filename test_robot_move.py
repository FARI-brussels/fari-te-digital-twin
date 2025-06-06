# %%
"""Basic IK

Simplest Inverse Kinematics Example using PyRoki.
"""

import time

import numpy as np
import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf
import zmq
import pyroki_snippets as pks
import struct

context = zmq.Context()
publisher = context.socket(zmq.PUB)
publisher.bind("tcp://0.0.0.0:5555")
def send_robot_data_fast(publisher, robot_id, joint_positions):
    # Pack: robot_id (4 bytes) + joint_positions (float32 array)
    joint_bytes = joint_positions.astype(np.float32).tobytes()
    message = struct.pack('I', robot_id) + joint_bytes
    print("sending message", message)
    publisher.send(message)

def main():
    """Main function for basic IK."""

    urdf = load_robot_description("panda_description")
    target_link_name = "panda_hand"

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)

    # Set up visualizer.
    server = viser.ViserServer()
    
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Create interactive controller with initial position.
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.61, 0.0, 0.56), wxyz=(0, 0, 1, 0)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    while True:
        # Solve IK.
        start_time = time.time()
        solution = pks.solve_ik(
            robot=robot,
            target_link_name=target_link_name,
            target_position=np.array(ik_target.position),
            target_wxyz=np.array(ik_target.wxyz),
        )

        # Update timing handle.
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        # Update visualizer.
        urdf_vis.update_cfg(solution)
        # Send solution to visualisation server.
        send_robot_data_fast(publisher, 1, solution)


if __name__ == "__main__":
    main()

# %%
