#!/usr/bin/env python3
"""Visualize two G1 robot retargeting results (baseline vs with_prior) side by side."""

import argparse
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

ROBOT_DOF = 29


def discover_sequences():
    """Find matching sequences across baseline and with_prior dirs."""
    sequences = {}
    for npz_file in sorted(BASELINE_DIR.glob("*_original.npz")):
        stem = npz_file.stem.replace("_original", "")
        prior_file = WITH_PRIOR_DIR / npz_file.name
        if prior_file.exists():
            sequences[stem] = {
                "baseline": npz_file,
                "with_prior": prior_file,
            }
    return sequences


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

    # Ground grid
    server.scene.add_grid(
        "/grid", width=100, height=100, position=(0.0, 0.0, 0.0),
        cell_size=4, cell_thickness=1, section_size=4, plane="xy",
    )

    # ---- State ----
    state = {
        "baseline": None,
        "prior": None,
        "num_frames": 0,
        "fps": 30,
    }

    def load_sequence(name):
        info = sequences[name]
        robot_offset = 1.5
        state["baseline"] = load_robot_npz(info["baseline"], x_offset=-robot_offset)
        state["prior"] = load_robot_npz(info["with_prior"], x_offset=robot_offset)
        state["num_frames"] = min(state["baseline"]["num_frames"], state["prior"]["num_frames"])
        state["fps"] = state["baseline"]["fps"]
        timestep_slider.max = max(0, state["num_frames"] - 1)
        timestep_slider.value = 0
        print(f"Loaded sequence '{name}': {state['num_frames']} frames, fps={state['fps']}")

    # ---- GUI ----
    seq_dropdown = server.gui.add_dropdown("Sequence", seq_names, initial_value=seq_names[0])
    playing = server.gui.add_checkbox("Playing", False)
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

        time.sleep(1.0 / state["fps"])


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
