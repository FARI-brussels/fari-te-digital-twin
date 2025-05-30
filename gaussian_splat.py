"""WebGL-based Gaussian splat rendering. This is still under developmentt."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import tyro
from plyfile import PlyData

import viser
from viser import transforms as tf
import pyroki as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

class SplatFile(TypedDict):
    """Data loaded from an antimatter15-style splat file."""

    centers: npt.NDArray[np.floating]
    """(N, 3)."""
    rgbs: npt.NDArray[np.floating]
    """(N, 3). Range [0, 1]."""
    opacities: npt.NDArray[np.floating]
    """(N, 1). Range [0, 1]."""
    covariances: npt.NDArray[np.floating]
    """(N, 3, 3)."""


def load_splat_file(splat_path: Path, center: bool = False) -> SplatFile:
    """Load an antimatter15-style splat file."""
    start_time = time.time()
    splat_buffer = splat_path.read_bytes()
    bytes_per_gaussian = (
        # Each Gaussian is serialized as:
        # - position (vec3, float32)
        3 * 4
        # - xyz (vec3, float32)
        + 3 * 4
        # - rgba (vec4, uint8)
        + 4
        # - ijkl (vec4, uint8), where 0 => -1, 255 => 1.
        + 4
    )
    assert len(splat_buffer) % bytes_per_gaussian == 0
    num_gaussians = len(splat_buffer) // bytes_per_gaussian

    # Reinterpret cast to dtypes that we want to extract.
    splat_uint8 = np.frombuffer(splat_buffer, dtype=np.uint8).reshape(
        (num_gaussians, bytes_per_gaussian)
    )
    scales = splat_uint8[:, 12:24].copy().view(np.float32)
    wxyzs = splat_uint8[:, 28:32] / 255.0 * 2.0 - 1.0
    Rs = tf.SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    centers = splat_uint8[:, 0:12].copy().view(np.float32)
    if center:
        centers -= np.mean(centers, axis=0, keepdims=True)
    print(
        f"Splat file with {num_gaussians=} loaded in {time.time() - start_time} seconds"
    )
    return {
        "centers": centers,
        # Colors should have shape (N, 3).
        "rgbs": splat_uint8[:, 24:27] / 255.0,
        "opacities": splat_uint8[:, 27:28] / 255.0,
        # Covariances should have shape (N, 3, 3).
        "covariances": covariances,
    }


def load_ply_file(ply_file_path: Path, center: bool = False) -> SplatFile:
    """Load Gaussians stored in a PLY file."""
    start_time = time.time()

    SH_C0 = 0.28209479177387814

    plydata = PlyData.read(ply_file_path)
    v = plydata["vertex"]
    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1))
    wxyzs = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
    colors = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    opacities = 1.0 / (1.0 + np.exp(-v["opacity"][:, None]))

    Rs = tf.SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    if center:
        positions -= np.mean(positions, axis=0, keepdims=True)

    num_gaussians = len(v)
    print(
        f"PLY file with {num_gaussians=} loaded in {time.time() - start_time} seconds"
    )
    return {
        "centers": positions,
        "rgbs": colors,
        "opacities": opacities,
        "covariances": covariances,
    }


def main(splat_paths: tuple[Path, ...]) -> None:
    server = viser.ViserServer()
    server.gui.configure_theme(dark_mode=True)
    gui_reset_up = server.gui.add_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = np.array([-0.56464173, 3.38609594, 0.0])
        client.camera.wxyz = np.array([0.0, 0.0, 0.0, 1.0])
        client.camera.look_at = np.array([3, 0.0, 0.0])
        client.camera.up_direction = np.array([0.0, 1.0, 0.0])

    @gui_reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.camera.position = np.array([-0.56464173, 3.38609594, 0.0])
        client.camera.wxyz = np.array([0.0, 0.0, 0.0, 1.0])
        client.camera.look_at = np.array([3, 0.0, 0.0])
        client.camera.up_direction = np.array([0.0, 1.0, 0.0])
        print(client.camera.position)
        print(client.camera.wxyz)
        print(client.camera.up_direction)

    frame = server.scene.add_frame("Add frame")
    grid = server.scene.add_grid("grid", plane="xz")
    light = server.scene.add_light_ambient("light", intensity=0.5)
    with open("demo_tables.glb", "rb") as f:
        stand_glb = f.read()
    server.scene.add_glb(f"/stand", stand_glb)
    for i, splat_path in enumerate(splat_paths):
        if splat_path.suffix == ".splat":
            splat_data = load_splat_file(splat_path, center=False)
        elif splat_path.suffix == ".ply":
            splat_data = load_ply_file(splat_path, center=False)
        else:
            raise SystemExit("Please provide a filepath to a .splat or .ply file.")
        """
        server.scene.add_transform_controls(f"/{i}")
        gs_handle = server.scene.add_gaussian_splats(
            f"/{i}/gaussian_splats",
            centers=splat_data["centers"],
            rgbs=splat_data["rgbs"],
            opacities=splat_data["opacities"],
            covariances=splat_data["covariances"],
        )
        """
        
        remove_button = server.gui.add_button(f"Remove splat object {i}")
        """
        @remove_button.on_click
        def _(_, gs_handle=gs_handle, remove_button=remove_button) -> None:
            gs_handle.remove()
            remove_button.remove()
        """

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main)