#!/usr/bin/env python3
"""Visualize two G1 robot retargeting results (baseline vs with_prior) side by side,
with the original LaFAN1 skeleton motion in the center."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
URDF_PATH = SCRIPT_DIR / "models" / "g1" / "g1_29dof.urdf"
BASELINE_DIR = SCRIPT_DIR / "demo_results_parallel" / "lafan_baseline"
WITH_PRIOR_DIR = SCRIPT_DIR / "demo_results_parallel" / "lafan_with_prior"
LAFAN1_PKG_DIR = SCRIPT_DIR / "ubisoft-laforge-animation-dataset"
BVH_DIR = LAFAN1_PKG_DIR / "lafan1" / "lafan1"

ROBOT_DOF = 29
SKELETON_COLOR = (50, 220, 130)  # bright green — easy to distinguish from pink/yellow robots
JOINT_RADIUS = 0.025  # meters

# Import LaFAN1 BVH loading utilities
sys.path.insert(0, str(LAFAN1_PKG_DIR))
from lafan1.extract import read_bvh
from lafan1.utils import quat_fk


def discover_sequences():
    """Find matching sequences across baseline and with_prior dirs."""
    sequences = {}
    for npz_file in sorted(BASELINE_DIR.glob("*_original.npz")):
        stem = npz_file.stem.replace("_original", "")
        prior_file = WITH_PRIOR_DIR / npz_file.name
        bvh_file = BVH_DIR / f"{stem}.bvh"
        if prior_file.exists():
            sequences[stem] = {
                "baseline": npz_file,
                "with_prior": prior_file,
                "bvh": bvh_file if bvh_file.exists() else None,
            }
    return sequences


def load_bvh_motion(bvh_path):
    """Load BVH file, compute global joint positions for all frames."""
    anim = read_bvh(str(bvh_path))
    _, global_pos = quat_fk(anim.quats, anim.pos, anim.parents)
    # Convert Y-up to Z-up: x_new = x, y_new = -z, z_new = y
    global_pos_zup = np.stack([
        global_pos[..., 0],
        -global_pos[..., 2],
        global_pos[..., 1],
    ], axis=-1).astype(np.float32)
    # BVH positions are in centimeters, convert to meters
    global_pos_zup /= 100.0
    # Y-axis offset to align skeleton with robots (adjust as needed)
    global_pos_zup[..., 1] += 0.3
    # Pre-compute bone index pairs for vectorized segment building
    parents = anim.parents
    parent_idx = []
    child_idx = []
    for i in range(len(parents)):
        if parents[i] >= 0:
            parent_idx.append(parents[i])
            child_idx.append(i)
    return {
        "global_pos": global_pos_zup,
        "parent_idx": np.array(parent_idx),
        "child_idx": np.array(child_idx),
        "num_frames": global_pos_zup.shape[0],
        "num_joints": global_pos_zup.shape[1],
    }


def load_robot_npz(npz_path, x_offset=0.0):
    """Load robot qpos from npz. qpos layout: root_pos(3) + root_quat_wxyz(4) + joints(29)."""
    data = np.load(npz_path, allow_pickle=True)
    qpos = data["qpos"].astype(np.float32)
    fps = int(data["fps"]) if "fps" in data else 30
    root_pos = qpos[:, 0:3].copy()
    root_quat_wxyz = qpos[:, 3:7].copy()
    joints = qpos[:, 7:7 + ROBOT_DOF].copy()
    if x_offset != 0.0:
        root_pos[:, 0] += x_offset
    return {
        "root_pos": root_pos,
        "root_quat_wxyz": root_quat_wxyz,
        "joints": joints,
        "num_frames": qpos.shape[0],
        "fps": fps,
    }


def main(args):
    sequences = discover_sequences()
    if not sequences:
        print("No matching sequences found.")
        print(f"  BASELINE_DIR: {BASELINE_DIR}  exists={BASELINE_DIR.exists()}")
        print(f"  WITH_PRIOR_DIR: {WITH_PRIOR_DIR}  exists={WITH_PRIOR_DIR.exists()}")
        return

    seq_names = sorted(sequences.keys())
    print(f"Found {len(seq_names)} sequence(s): {seq_names}")

    # Load URDF
    urdf = yourdfpy.URDF.load(str(URDF_PATH))

    # Start viser server
    server = viser.ViserServer(port=8088)
    server.scene.set_up_direction("+z")

    # Robot frames
    baseline_frame = server.scene.add_frame("/baseline", show_axes=False)
    baseline_urdf_vis = ViserUrdf(server, urdf, root_node_name="/baseline", mesh_color_override=(237, 194, 199))

    prior_frame = server.scene.add_frame("/with_prior", show_axes=False)
    prior_urdf_vis = ViserUrdf(server, urdf, root_node_name="/with_prior", mesh_color_override=(237, 236, 194))

    # Background settings: ground grid + shadow-receiving box
    box_size = 100
    ground_grid = server.scene.add_grid(
        "/grid", width=box_size, height=box_size, position=(0.0, -40.0, 0.01),
        cell_size=4, cell_thickness=1, section_size=4,
        plane="xy", shadow_opacity=0.0,
    )
    bg_box = server.scene.add_box(
        name="/box", dimensions=(box_size, box_size, box_size),
        position=(0.0, -40.0, -box_size // 2),
        color=(170, 170, 150), material='standard', side='double',
        cast_shadow=False, flat_shading=True, receive_shadow=True,
        visible=True,
    )

    # Skeleton visualization (LaFAN1 original motion)
    _n_bones = 21  # 22 joints, root has no parent
    _n_joints = 22
    skeleton_lines_handle = server.scene.add_line_segments(
        "/lafan_skeleton",
        points=np.zeros((_n_bones, 2, 3), dtype=np.float32),
        colors=np.full((_n_bones, 2, 3), np.array(SKELETON_COLOR, dtype=np.uint8)),
        line_width=8.0,
    )
    # Use individual spheres for joints — much more visible than point cloud
    joint_sphere_handles = []
    for j in range(_n_joints):
        h = server.scene.add_icosphere(
            f"/lafan_joint/{j}",
            radius=JOINT_RADIUS,
            color=SKELETON_COLOR,
            position=(0.0, 0.0, 0.0),
        )
        h.visible = False
        joint_sphere_handles.append(h)
    skeleton_lines_handle.visible = False

    # ---- State ----
    state = {
        "baseline": None,
        "prior": None,
        "lafan": None,
        "num_frames": 0,
        "fps": 30,
    }

    def load_sequence(name):
        info = sequences[name]
        robot_offset = 1.5
        state["baseline"] = load_robot_npz(info["baseline"], x_offset=-robot_offset)
        state["prior"] = load_robot_npz(info["with_prior"], x_offset=robot_offset)

        # Load LaFAN1 BVH
        if info.get("bvh") is not None:
            state["lafan"] = load_bvh_motion(info["bvh"])
            skeleton_lines_handle.visible = True
            for h in joint_sphere_handles:
                h.visible = True
        else:
            state["lafan"] = None
            skeleton_lines_handle.visible = False
            for h in joint_sphere_handles:
                h.visible = False
            print(f"  Warning: no BVH file for sequence '{name}'")

        frame_counts = [state["baseline"]["num_frames"], state["prior"]["num_frames"]]
        if state["lafan"] is not None:
            frame_counts.append(state["lafan"]["num_frames"])
        state["num_frames"] = min(frame_counts)
        state["fps"] = state["baseline"]["fps"]
        fps_slider.value = state["fps"]
        timestep_slider.max = max(0, state["num_frames"] - 1)
        timestep_slider.value = 0
        print(f"Loaded sequence '{name}': {state['num_frames']} frames, fps={state['fps']}")

    # ---- GUI ----
    seq_dropdown = server.gui.add_dropdown("Sequence", seq_names, initial_value=seq_names[0])
    playing = server.gui.add_checkbox("Playing", False)
    fps_slider = server.gui.add_slider("FPS", 1, 120, 1, 30)
    timestep_slider = server.gui.add_slider("Timestep", 0, 100, 1, 0)

    @seq_dropdown.on_update
    def _(_):
        load_sequence(seq_dropdown.value)

    # Camera controls
    with server.gui.add_folder("Camera Settings"):
        cam_pos_x = server.gui.add_number("Camera X", initial_value=0.0, step=0.1)
        cam_pos_y = server.gui.add_number("Camera Y", initial_value=-7.1, step=0.1)
        cam_pos_z = server.gui.add_number("Camera Z", initial_value=1.25, step=0.01)
        cam_toward_x = server.gui.add_number("Camera Toward X", initial_value=0.0, step=0.1)
        cam_toward_y = server.gui.add_number("Camera Toward Y", initial_value=0.0, step=0.1)
        cam_toward_z = server.gui.add_number("Camera Toward Z", initial_value=1.05, step=0.01)
        fov = server.gui.add_number("Camera FOV", initial_value=45.0, step=1.0)
        set_cam_button = server.gui.add_button("Set Camera")

        def set_camera(_=None):
            clients = list(server.get_clients().values())
            if not clients:
                return
            client = clients[0]
            client.camera.position = np.array([cam_pos_x.value, cam_pos_y.value, cam_pos_z.value])
            client.camera.look_at = np.array([cam_toward_x.value, cam_toward_y.value, cam_toward_z.value])
            client.camera.fov = fov.value * np.pi / 360.0

        for ctrl in [cam_pos_x, cam_pos_y, cam_pos_z, cam_toward_x, cam_toward_y, cam_toward_z, fov]:
            ctrl.on_update(set_camera)
        set_cam_button.on_click(set_camera)

    # Background position controls
    with server.gui.add_folder("Background Settings"):
        box_x = server.gui.add_number(label="Box position X", initial_value=0.0, step=5.0)
        box_y = server.gui.add_number(label="Box position Y", initial_value=-40.0, step=5.0)
        current_z = [0.0]  # mutable container for nonlocal-like access

        box_z = server.gui.add_number(label="Box position Z", initial_value=0.0, step=0.01)

        def set_box_pos_xy(_):
            bg_box.position = (box_x.value, box_y.value, bg_box.position[2])
            ground_grid.position = (box_x.value, box_y.value, ground_grid.position[2])

        for ctrl in [box_x, box_y]:
            ctrl.on_update(set_box_pos_xy)

        @box_z.on_update
        def _(_):
            delta_z = box_z.value - current_z[0]
            current_z[0] = box_z.value
            bg_box.position = (bg_box.position[0], bg_box.position[1], bg_box.position[2] + delta_z)
            ground_grid.position = (ground_grid.position[0], ground_grid.position[1], ground_grid.position[2] + delta_z)
            cam_pos_z.value = cam_pos_z.value + delta_z
            cam_toward_z.value = cam_toward_z.value + delta_z

    # Load initial sequence
    initial_seq = seq_names[0]
    if args.seq and args.seq in sequences:
        initial_seq = args.seq
        seq_dropdown.value = initial_seq
    elif args.seq:
        print(f"Warning: sequence '{args.seq}' not found, using '{initial_seq}'")
    load_sequence(initial_seq)

    # ---- Main loop ----
    while True:
        if state["num_frames"] == 0:
            time.sleep(0.1)
            continue

        with server.atomic():
            if playing.value:
                timestep_slider.value = (timestep_slider.value + 1) % state["num_frames"]

            t = int(timestep_slider.value)

            # Baseline robot
            bl = state["baseline"]
            baseline_frame.position = bl["root_pos"][t]
            baseline_frame.wxyz = bl["root_quat_wxyz"][t]
            baseline_urdf_vis.update_cfg(bl["joints"][t])

            # With-prior robot
            pr = state["prior"]
            prior_frame.position = pr["root_pos"][t]
            prior_frame.wxyz = pr["root_quat_wxyz"][t]
            prior_urdf_vis.update_cfg(pr["joints"][t])

            # LaFAN1 skeleton
            if state["lafan"] is not None:
                lafan = state["lafan"]
                gpos = lafan["global_pos"][t]  # (22, 3)
                segments = np.stack(
                    [gpos[lafan["parent_idx"]], gpos[lafan["child_idx"]]], axis=1
                )  # (21, 2, 3)
                skeleton_lines_handle.points = segments
                for j, h in enumerate(joint_sphere_handles):
                    h.position = gpos[j]

        time.sleep(1.0 / max(1, fps_slider.value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=str, default=None, help="Sequence name, e.g. 'walk1_subject1'")
    parser.add_argument("--list", action="store_true", help="List all available sequences and exit")
    args = parser.parse_args()

    if args.list:
        seqs = discover_sequences()
        print(f"Available sequences ({len(seqs)}):")
        for name in sorted(seqs.keys()):
            print(f"  {name}")
    else:
        main(args)
