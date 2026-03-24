#!/usr/bin/env python3
"""Visualize human SMPL-X mesh + two G1 robot retargeting results (baseline vs with_prior) side by side.

LAFAN joint positions are fitted to SMPL-X parameters via IK, then visualized as mesh.
Fitted results are cached to disk for fast subsequent loads.
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import smplx
import viser
import yourdfpy
from viser.extras import ViserUrdf
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
URDF_PATH = SCRIPT_DIR / "models" / "g1" / "g1_29dof.urdf"
BASELINE_DIR = SCRIPT_DIR / "demo_results_parallel" / "lafan_baseline"
WITH_PRIOR_DIR = SCRIPT_DIR / "demo_results_parallel" / "lafan_with_prior"
HUMAN_DIR = SCRIPT_DIR / "demo_data" / "lafan"
SMPLX_MODEL_DIR = SCRIPT_DIR / "models" / "human_body"
CACHE_DIR = SCRIPT_DIR / "demo_data" / "lafan_smplx_cache"

ROBOT_DOF = 29

# ---------------------------------------------------------------------------
# LAFAN joint index → SMPL-X joint index mapping (both 22 joints)
# ---------------------------------------------------------------------------
# LAFAN:  0:Hips 1:RUpLeg 2:RLeg 3:RFoot 4:RToe 5:LUpLeg 6:LLeg 7:LFoot 8:LToe
#         9:Spine 10:Spine1 11:Spine2 12:Neck 13:Head
#         14:RShoulder 15:RArm 16:RForeArm 17:RHand
#         18:LShoulder 19:LArm 20:LForeArm 21:LHand
# SMPLX:  0:Pelvis 1:L_Hip 2:R_Hip 3:Spine1 4:L_Knee 5:R_Knee
#         6:Spine2 7:L_Ankle 8:R_Ankle 9:Spine3 10:L_Foot 11:R_Foot
#         12:Neck 13:L_Collar 14:R_Collar 15:Head 16:L_Shoulder 17:R_Shoulder
#         18:L_Elbow 19:R_Elbow 20:L_Wrist 21:R_Wrist
SMPLX_INDICES_FOR_LAFAN = [0, 2, 5, 8, 11, 1, 4, 7, 10, 3, 6, 9, 12, 15, 14, 17, 19, 21, 13, 16, 18, 20]


def discover_sequences():
    """Find matching sequences across baseline, with_prior and human dirs."""
    print(f"[DEBUG] SCRIPT_DIR: {SCRIPT_DIR}")
    print(f"[DEBUG] BASELINE_DIR: {BASELINE_DIR}  exists={BASELINE_DIR.exists()}")
    print(f"[DEBUG] WITH_PRIOR_DIR: {WITH_PRIOR_DIR}  exists={WITH_PRIOR_DIR.exists()}")
    print(f"[DEBUG] HUMAN_DIR: {HUMAN_DIR}  exists={HUMAN_DIR.exists()}")

    if BASELINE_DIR.exists():
        baseline_files = list(BASELINE_DIR.glob("*_original.npz"))
        print(f"[DEBUG] Baseline *_original.npz files: {[f.name for f in baseline_files]}")
        if not baseline_files:
            all_files = list(BASELINE_DIR.iterdir())
            print(f"[DEBUG] All files in baseline dir: {[f.name for f in all_files[:20]]}")

    if HUMAN_DIR.exists():
        human_files = list(HUMAN_DIR.glob("*.npy"))
        print(f"[DEBUG] Human .npy files: {[f.name for f in human_files[:20]]}")

    sequences = {}
    for npz_file in sorted(BASELINE_DIR.glob("*_original.npz")):
        stem = npz_file.stem.replace("_original", "")
        human_file = HUMAN_DIR / f"{stem}.npy"
        prior_file = WITH_PRIOR_DIR / npz_file.name
        if human_file.exists() and prior_file.exists():
            sequences[stem] = {
                "baseline": npz_file,
                "with_prior": prior_file,
                "human": human_file,
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


# ---------------------------------------------------------------------------
# SMPL-X IK fitting
# ---------------------------------------------------------------------------
def fit_smplx_to_lafan(lafan_joints_np, smplx_model, device="cpu", batch_size=64, num_iters=150):
    """Fit SMPL-X parameters to LAFAN joint positions via optimization.

    Args:
        lafan_joints_np: (N, 22, 3) LAFAN joint positions
        smplx_model: smplx body model
        device: torch device
        batch_size: number of frames to optimize simultaneously
        num_iters: optimization iterations per batch

    Returns:
        dict with 'vertices' (N, V, 3) and 'faces' (F, 3)
    """
    N = lafan_joints_np.shape[0]
    smplx_idx = torch.tensor(SMPLX_INDICES_FOR_LAFAN, dtype=torch.long, device=device)
    target_all = torch.tensor(lafan_joints_np, dtype=torch.float32, device=device)  # (N, 22, 3)

    all_vertices = []
    faces = smplx_model.faces.astype(np.int32)

    # Fixed betas (neutral body shape)
    betas = torch.zeros(1, 10, dtype=torch.float32, device=device)

    prev_global_orient = torch.zeros(1, 3, device=device)
    prev_body_pose = torch.zeros(1, 63, device=device)

    for start in tqdm(range(0, N, batch_size), desc="Fitting SMPL-X"):
        end = min(start + batch_size, N)
        B = end - start
        target_batch = target_all[start:end]  # (B, 22, 3)

        # Initialize from previous frame's result
        global_orient = prev_global_orient.expand(B, -1).clone().detach().requires_grad_(True)
        body_pose = prev_body_pose.expand(B, -1).clone().detach().requires_grad_(True)
        transl = target_batch[:, 0, :].clone().detach().requires_grad_(True)  # init from Hips pos

        optimizer = torch.optim.Adam([global_orient, body_pose, transl], lr=0.02)

        for it in range(num_iters):
            optimizer.zero_grad()
            output = smplx_model(
                betas=betas.expand(B, -1),
                global_orient=global_orient,
                body_pose=body_pose,
                transl=transl,
                left_hand_pose=torch.zeros(B, 45, device=device),
                right_hand_pose=torch.zeros(B, 45, device=device),
                jaw_pose=torch.zeros(B, 3, device=device),
                leye_pose=torch.zeros(B, 3, device=device),
                reye_pose=torch.zeros(B, 3, device=device),
                expression=torch.zeros(B, 10, device=device),
                return_full_pose=True,
            )
            pred_joints = output.joints[:, smplx_idx, :]  # (B, 22, 3)
            loss = torch.mean((pred_joints - target_batch) ** 2)

            # Temporal smoothness regularization
            if B > 1:
                pose_diff = body_pose[1:] - body_pose[:-1]
                orient_diff = global_orient[1:] - global_orient[:-1]
                loss = loss + 0.001 * (torch.mean(pose_diff ** 2) + torch.mean(orient_diff ** 2))

            loss.backward()
            optimizer.step()

            # Reduce LR halfway
            if it == num_iters // 2:
                for pg in optimizer.param_groups:
                    pg['lr'] = 0.005

        # Store last frame's params as init for next batch
        prev_global_orient = global_orient[-1:].detach()
        prev_body_pose = body_pose[-1:].detach()

        # Get final vertices
        with torch.no_grad():
            output = smplx_model(
                betas=betas.expand(B, -1),
                global_orient=global_orient,
                body_pose=body_pose,
                transl=transl,
                left_hand_pose=torch.zeros(B, 45, device=device),
                right_hand_pose=torch.zeros(B, 45, device=device),
                jaw_pose=torch.zeros(B, 3, device=device),
                leye_pose=torch.zeros(B, 3, device=device),
                reye_pose=torch.zeros(B, 3, device=device),
                expression=torch.zeros(B, 10, device=device),
                return_full_pose=True,
            )
            all_vertices.append(output.vertices.cpu().numpy())

    all_vertices = np.concatenate(all_vertices, axis=0)  # (N, V, 3)
    return {"vertices": all_vertices, "faces": faces}


def _create_smplx_model(device="cpu"):
    """Create SMPL-X model. Same pattern as viser_for_human_robot.py / src/utils.py.

    If pkl loading fails (smplx version bug), auto-converts pkl→npz and retries.
    """
    pkl_path = SMPLX_MODEL_DIR / "smplx" / "SMPLX_NEUTRAL.pkl"
    npz_path = SMPLX_MODEL_DIR / "smplx" / "SMPLX_NEUTRAL.npz"

    # Try loading pkl directly (same as original viser_for_human_robot.py)
    try:
        model = smplx.create(
            str(SMPLX_MODEL_DIR), 'smplx',
            gender='NEUTRAL', use_pca=False, ext='pkl',
        ).to(device)
        return model
    except Exception:
        pass

    # Fallback: convert pkl→npz with pickle.load(encoding='latin1'), then load npz
    if not npz_path.exists():
        print(f"Converting {pkl_path.name} → {npz_path.name} for compatibility...")
        with open(str(pkl_path), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        save_dict = {}
        for k, v in data.items():
            if hasattr(v, 'toarray'):  # scipy sparse
                save_dict[k] = np.array(v.toarray())
            elif isinstance(v, np.ndarray):
                save_dict[k] = v
            else:
                try:
                    save_dict[k] = np.array(v)
                except Exception:
                    pass
        np.savez(str(npz_path), **save_dict)

    model = smplx.create(
        str(SMPLX_MODEL_DIR), 'smplx',
        gender='NEUTRAL', use_pca=False, ext='npz',
    ).to(device)
    return model


def get_or_fit_smplx(seq_name, human_npy_path, device="cpu"):
    """Load cached SMPL-X fit or compute and cache it."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{seq_name}.npz"

    if cache_file.exists():
        print(f"Loading cached SMPL-X fit for '{seq_name}'...")
        data = np.load(cache_file, allow_pickle=True)
        return {"vertices": data["vertices"], "faces": data["faces"]}

    print(f"Fitting SMPL-X to LAFAN joints for '{seq_name}' (this may take a while)...")
    lafan_joints = np.load(human_npy_path, allow_pickle=True).astype(np.float32)

    smplx_model = _create_smplx_model(device)
    smplx_model.eval()

    result = fit_smplx_to_lafan(lafan_joints, smplx_model, device=device)

    np.savez_compressed(str(cache_file), vertices=result["vertices"], faces=result["faces"])
    print(f"Cached SMPL-X fit to {cache_file}")
    return result


# ---------------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------------
def main(args):
    sequences = discover_sequences()
    if not sequences:
        print("No matching sequences found. Check directory paths.")
        return

    seq_names = sorted(sequences.keys())
    print(f"Found {len(seq_names)} sequence(s): {seq_names}")

    # Detect device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

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

    # Human mesh — initial dummy (will be replaced on sequence load)
    dummy_verts = np.zeros((3, 3), dtype=np.float32)
    dummy_faces = np.array([[0, 1, 2]], dtype=np.int32)
    body_handle = server.scene.add_mesh_simple(
        "/human",
        vertices=dummy_verts,
        faces=dummy_faces,
        color=(194, 223, 237),
        wireframe=False,
        side='double',
    )

    # Ground grid
    server.scene.add_grid(
        "/grid", width=100, height=100, position=(0.0, 0.0, 0.0),
        cell_size=4, cell_thickness=1, section_size=4, plane="xy",
    )

    # ---- State ----
    state = {
        "baseline": None,
        "prior": None,
        "human_vertices": None,
        "human_faces": None,
        "num_frames": 0,
        "fps": 30,
    }

    def load_sequence(name):
        info = sequences[name]
        robot_offset = 1.5
        state["baseline"] = load_robot_npz(info["baseline"], x_offset=-robot_offset)
        state["prior"] = load_robot_npz(info["with_prior"], x_offset=robot_offset)

        # Fit or load cached SMPL-X
        human_data = get_or_fit_smplx(name, info["human"], device=device)
        state["human_vertices"] = human_data["vertices"]
        state["human_faces"] = human_data["faces"]

        state["num_frames"] = min(
            state["baseline"]["num_frames"],
            state["prior"]["num_frames"],
            state["human_vertices"].shape[0],
        )
        state["fps"] = state["baseline"]["fps"]
        timestep_slider.max = max(0, state["num_frames"] - 1)
        timestep_slider.value = 0

        # Update mesh faces (static)
        body_handle.faces = state["human_faces"]
        body_handle.vertices = state["human_vertices"][0]

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

            # Human mesh
            body_handle.vertices = state["human_vertices"][t]

        time.sleep(1.0 / state["fps"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=str, default=None, help="Sequence name to visualize, e.g. 'walk1_subject1'")
    parser.add_argument("--list", action="store_true", help="List all available sequences and exit")
    args = parser.parse_args()

    if args.list:
        seqs = discover_sequences()
        print(f"Available sequences ({len(seqs)}):")
        for name in sorted(seqs.keys()):
            print(f"  {name}")
    else:
        main(args)
