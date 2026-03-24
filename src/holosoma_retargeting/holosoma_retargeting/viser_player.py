#!/usr/bin/env python3
# viser_player.py
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import tyro
import viser  # type: ignore[import-not-found]  # pip install viser
import yourdfpy  # type: ignore[import-untyped]  # pip install yourdfpy
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

src_root = Path(__file__).resolve().parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
from holosoma_retargeting.config_types.viser import ViserConfig  # noqa: E402
from holosoma_retargeting.src.viser_utils import create_motion_control_sliders  # noqa: E402


def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    # expected: qpos [T, ?], and optional fps
    qpos = data["qpos"]
    fps = int(data["fps"]) if "fps" in data else 30
    return qpos, fps


def make_player(
    config: ViserConfig,
    qpos: np.ndarray,
    fps: int | None = None,
):
    """
    qpos layout (MuJoCo order):
      [0:3]   robot base position (xyz)
      [3:7]   robot base quat (wxyz)
      [7:7+R] robot joint positions (R = actuated dof)
      [end-7:end-4] (optional) object position (xyz)
      [end-4:end]   (optional) object quat (wxyz)

    We'll infer R from the robot URDF's actuated joints in ViserUrdf.
    """
    server = viser.ViserServer()

    # Root frames
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    object_root = server.scene.add_frame("/object", show_axes=False)

    # URDFs (using yourdfpy so meshes show up)
    robot_urdf_y = yourdfpy.URDF.load(config.robot_urdf, load_meshes=True, build_scene_graph=True)
    vr = ViserUrdf(server, urdf_or_path=robot_urdf_y, root_node_name="/robot")

    vo = None
    if config.object_urdf:
        object_urdf_y = yourdfpy.URDF.load(config.object_urdf, load_meshes=True, build_scene_graph=True)
        vo = ViserUrdf(server, urdf_or_path=object_urdf_y, root_node_name="/object")

    # A tiny grid
    server.scene.add_grid("/grid", width=config.grid_width, height=config.grid_height, position=(0.0, 0.0, 0.0))

    # Figure robot DOF from actuated limits in ViserUrdf
    joint_limits = vr.get_actuated_joint_limits()
    robot_dof = len(joint_limits)

    # Use fps from config if not provided, otherwise use the one from npz file
    actual_fps = fps if fps is not None else config.fps

    # Set initial mesh visibility
    vr.show_visual = config.show_meshes
    if vo is not None:
        vo.show_visual = config.show_meshes

    # ---------- Additional GUI controls (mesh visibility) ----------
    with server.gui.add_folder("Display"):
        show_meshes_cb = server.gui.add_checkbox("Show meshes", initial_value=config.show_meshes)

    @show_meshes_cb.on_update
    def _(_):
        vr.show_visual = bool(show_meshes_cb.value)
        if vo is not None:
            vo.show_visual = bool(show_meshes_cb.value)

    # ---------- Camera controls ----------
    # Design: GUI numbers are read-only displays that track the browser camera.
    # To set camera programmatically: type values then click "Apply to Camera",
    # or check "Lock Camera" to continuously enforce GUI values.
    cam_state = {"lock": False}

    with server.gui.add_folder("Camera Settings"):
        cam_pos_x = server.gui.add_number("Camera X", initial_value=0.0, step=0.1, disabled=True)
        cam_pos_y = server.gui.add_number("Camera Y", initial_value=0.0, step=0.1, disabled=True)
        cam_pos_z = server.gui.add_number("Camera Z", initial_value=0.0, step=0.01, disabled=True)
        cam_toward_x = server.gui.add_number("Look-at X", initial_value=0.0, step=0.1, disabled=True)
        cam_toward_y = server.gui.add_number("Look-at Y", initial_value=0.0, step=0.1, disabled=True)
        cam_toward_z = server.gui.add_number("Look-at Z", initial_value=0.0, step=0.01, disabled=True)
        cam_fov_disp = server.gui.add_number("FOV (deg)", initial_value=45.0, step=1.0, disabled=True)
        snap_button = server.gui.add_button("Snapshot (read current)")

        @snap_button.on_click
        def _(_):
            """Unlock fields so user can edit, pre-filled with current camera."""
            for ctrl in [cam_pos_x, cam_pos_y, cam_pos_z, cam_toward_x, cam_toward_y, cam_toward_z, cam_fov_disp]:
                ctrl.disabled = False

        apply_button = server.gui.add_button("Apply to Camera")

        @apply_button.on_click
        def _(_):
            """Push GUI values to browser camera once."""
            clients = list(server.get_clients().values())
            for client in clients:
                client.camera.position = np.array([cam_pos_x.value, cam_pos_y.value, cam_pos_z.value])
                client.camera.look_at = np.array([cam_toward_x.value, cam_toward_y.value, cam_toward_z.value])
                client.camera.fov = cam_fov_disp.value * np.pi / 180.0
            # Re-lock the fields after applying
            for ctrl in [cam_pos_x, cam_pos_y, cam_pos_z, cam_toward_x, cam_toward_y, cam_toward_z, cam_fov_disp]:
                ctrl.disabled = True

        lock_cam_cb = server.gui.add_checkbox("Lock Camera", initial_value=False)

        @lock_cam_cb.on_update
        def _(_):
            cam_state["lock"] = bool(lock_cam_cb.value)

        def read_camera_to_gui():
            """Read browser camera state into GUI number displays."""
            clients = list(server.get_clients().values())
            if not clients:
                return
            client = clients[0]
            pos = client.camera.position
            look = client.camera.look_at
            fov_val = client.camera.fov
            if pos is None or look is None:
                return
            cam_pos_x.value = round(float(pos[0]), 3)
            cam_pos_y.value = round(float(pos[1]), 3)
            cam_pos_z.value = round(float(pos[2]), 3)
            cam_toward_x.value = round(float(look[0]), 3)
            cam_toward_y.value = round(float(look[1]), 3)
            cam_toward_z.value = round(float(look[2]), 3)
            if fov_val is not None:
                cam_fov_disp.value = round(float(fov_val) * 180.0 / np.pi, 1)

        def apply_camera():
            """Push GUI values to all browser clients."""
            clients = list(server.get_clients().values())
            for client in clients:
                client.camera.position = np.array([cam_pos_x.value, cam_pos_y.value, cam_pos_z.value])
                client.camera.look_at = np.array([cam_toward_x.value, cam_toward_y.value, cam_toward_z.value])
                client.camera.fov = cam_fov_disp.value * np.pi / 180.0

    # ---------- Use reusable motion control sliders from viser_utils ----------
    create_motion_control_sliders(
        server=server,
        viser_robot=vr,
        robot_base_frame=robot_root,
        motion_sequence=qpos,
        robot_dof=robot_dof,
        viser_object=vo if config.assume_object_in_qpos else None,
        object_base_frame=object_root if config.assume_object_in_qpos else None,
        contains_object_in_qpos=config.assume_object_in_qpos,
        initial_fps=actual_fps,
        initial_interp_mult=config.visual_fps_multiplier,
        loop=config.loop,
    )
    n_frames = int(qpos.shape[0])
    print(
        f"[viser_player] Loaded {n_frames} frames | robot_dof={robot_dof} | "
        f"object={'yes' if (config.object_urdf and config.assume_object_in_qpos) else 'no'}"
    )
    print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")
    return server, cam_state, apply_camera, read_camera_to_gui


def main(cfg: ViserConfig) -> None:
    """Main function for viser player."""
    qpos, fps = load_npz(cfg.qpos_npz)
    server, cam_state, apply_camera, read_camera_to_gui = make_player(
        config=cfg,
        qpos=qpos,
        fps=fps,
    )

    # Main loop: camera sync
    while True:
        if cam_state["lock"]:
            apply_camera()
        else:
            read_camera_to_gui()
        time.sleep(0.15)


if __name__ == "__main__":
    cfg = tyro.cli(ViserConfig)
    main(cfg)
