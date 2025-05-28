# %%
import time
import numpy as np
import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf
import pyroki_snippets as pks
import farizeromq as fzmq
import zmq

context = zmq.Context()
subscriber = context.socket(zmq.SUB)

# Connect to multiple publishers with different IPs
subscriber.connect("tcp://192.168.1.100:5555")


def main():
    """Main function for basic IK."""

    urdf = load_robot_description("panda_description")
    target_link_name = "panda_hand"

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)

    # Set up visualizer.
    server = viser.ViserServer()
    
    server.scene.add_grid("/ground", width=2, height=2)
    with open("stand.glb", "rb") as f:
        glb_data = f.read()
    server.scene.add_glb("/stand_1", glb_data)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    
    # Create interactive controller with initial position.
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.61, 0.0, 0.56), wxyz=(0, 0, 1, 0)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
    while True:
        message = subscriber.recv_string()
        print(f"Received: {message}")
        urdf_vis.update_cfg(message)

if __name__ == "__main__":
    main()
# %%
