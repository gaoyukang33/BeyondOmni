import time
from pathlib import Path
from typing import Tuple, TypedDict
from PIL import Image

import numpy as np
import os
import viser
from viser.extras import ViserUrdf
import yourdfpy

import torch
from general_motion_retargeting.nrdf_net import NRDF_Adapter
from tqdm import tqdm
import pickle
import smplx
from scipy.spatial.transform import Rotation as R


nrdf_model = NRDF_Adapter('cpu')
# state_dict = torch.load('/home/CONNECT/ygu425/Grain/code/MimicKit/prior_ckpts/nrdf_epoch_70.pt')['model']
state_dict = torch.load('/media/bic/77223f40-ef66-482e-aa9e-ad9ea5a67660/formal_exp/GMR/nrdf_upsample_with_phuma_epoch50.pt')['model']
nrdf_model.load_state_dict(state_dict)
nrdf_model.train(False)

def load_robot_motion(robot_motion_file, root_position_offset=None):
    with open(robot_motion_file, "rb") as filestream:
        in_dict = pickle.load(filestream)
        # print(in_dict.keys())
        frames = np.array(in_dict['frames'], dtype=np.float32)
        root_pos = frames[..., 0:3]
        root_rot = frames[..., 3:6]
        joint_dof = frames[..., 6:]
    
    root_rot = R.from_rotvec(root_rot).as_quat()
    num_timesteps = joint_dof.shape[0]
    joints = joint_dof

    # NRDF pre-calculate
    with torch.no_grad():
        nrdf_value = nrdf_model(torch.from_numpy(joints)).numpy()
    
    if root_position_offset is not None:
        root_pos[:, 0] += root_position_offset

    robot_dict = {
        "joints": joints,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "num_timesteps": num_timesteps,
        "nrdf_value": nrdf_value
    }
    return robot_dict

def load_human_motion(smpl_path, model_dir):
    model = smplx.create(
        str(model_dir),
        'smplx',
        gender='NEUTRAL',
        use_pca=False,
        ext = 'pkl',
    )
    smplx_data = np.load(smpl_path, allow_pickle=True)
    body_pose = torch.tensor(smplx_data["pose_body"]).float() # (N, 63)
    num_frames = body_pose.shape[0]
    smplx_output = model(
        betas=torch.tensor(smplx_data["betas"]).float().view(1, -1), # (16,)
        global_orient=torch.tensor(smplx_data["root_orient"]).float(), # (N, 3)
        body_pose=torch.tensor(smplx_data["pose_body"]).float(), # (N, 63)
        transl=torch.tensor(smplx_data["trans"]).float(), # (N, 3)
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        # expression=torch.zeros(num_frames, 10).float(),
        return_full_pose=True,
    )
    all_vertices = smplx_output.vertices.detach().cpu().numpy()
    root_pos = smplx_output.transl.detach().cpu().numpy()   # [frames, 3]
    top_k = 100
    min_z_per_frame = np.min(all_vertices[:, :, 2], axis=1)
    sorted_min_z = np.sort(min_z_per_frame)
    if num_frames < top_k:
        top_k = num_frames
    lowest_k_mins = sorted_min_z[10:top_k+10]
    ground_level_estimate = np.mean(lowest_k_mins)
    all_vertices[:, :, 2] -= ground_level_estimate

    F = model.faces.astype(np.int32)  

    human_dict = {
        'vertices': all_vertices,
        'faces': F,
        'transl': root_pos,
    }
    return human_dict

def main():    
    urdf = yourdfpy.URDF.load(
        "/media/bic/77223f40-ef66-482e-aa9e-ad9ea5a67660/LAFAN1_Retargeting_Dataset/robot_description/g1/g1_29dof_rev_1_0.urdf",
        mesh_dir="/media/bic/77223f40-ef66-482e-aa9e-ad9ea5a67660/LAFAN1_Retargeting_Dataset/robot_description/g1/meshes",
    )
    case_name = "Olivia_Miserable_C3D_stageii_1700_1900"
    record_frame_dir = os.path.join('./record_frames', case_name, 'images')
    os.makedirs(record_frame_dir, exist_ok=True)

    smpl_path = f'./data/Amass_sample/{case_name}.npz'
    body_path = '/media/bic/77223f40-ef66-482e-aa9e-ad9ea5a67660/formal_exp/GMR/assets/body_models'
    human_motion_dict = load_human_motion(smpl_path, body_path)

    # nrdf_motion_file = f"./data/Ours_res/{case_name}.pkl"
    # gmr_motion_file = f"./data/GMR_res/{case_name}.pkl"
    # nrdf_motion_file = f"./record_pkl_amassretargeting/Ours/{case_name}/ref.pkl"
    nrdf_motion_file = f"./record_pkl_amassretargeting/Ours/{case_name}/char.pkl"
    
    # gmr_motion_file = f"./record_pkl_amassretargeting/GMR/{case_name}/ref.pkl"
    gmr_motion_file = f"./record_pkl_amassretargeting/GMR/{case_name}/char.pkl"

    robot_offset = 1.2
    ours_res = load_robot_motion(nrdf_motion_file, robot_offset)
    gmr_res = load_robot_motion(gmr_motion_file, -robot_offset)

    server = viser.ViserServer(port=8088)
    nrdf_frame = server.scene.add_frame("/nrdf", show_axes=False)
    ours_urdf_vis = ViserUrdf(server, urdf, root_node_name="/nrdf", mesh_color_override=(237, 236, 194))

    gmr_frame = server.scene.add_frame("/gmr", show_axes=False)
    gmr_urdf_vis = ViserUrdf(server, urdf, root_node_name="/gmr", mesh_color_override=(237, 194, 199))
    # gmr_urdf_vis = ViserUrdf(server, urdf, root_node_name="/gmr")

    body_handle = server.scene.add_mesh_simple(
        "/human",
        vertices=human_motion_dict['vertices'][0],
        faces=human_motion_dict['faces'],
        color=(194, 223, 237),
        wireframe=False,
        side = 'double'
        # wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,  # rotate for a typical upright view
    )
    
    server.scene.set_up_direction("+z")
    playing = server.gui.add_checkbox("playing", False)
    recording = server.gui.add_checkbox("recording", False)
    track_x = server.gui.add_checkbox("track_x", False)
    track_y = server.gui.add_checkbox("track_y", False)

    timestep_slider = server.gui.add_slider("timestep", 0, ours_res['num_timesteps'] - 1, 1, 0)

    with server.gui.add_folder('Camera Settings'):
        cam_pos_x = server.gui.add_number("Camera X", initial_value=0.0, step=0.1)
        cam_pos_y = server.gui.add_number("Camera Y", initial_value=-7.1, step=0.1)
        cam_pos_z = server.gui.add_number("Camera Z", initial_value=1.25, step=0.01)
        cam_toward_x = server.gui.add_number("Camera Toward X", initial_value=0.0, step=0.1)
        cam_toward_y = server.gui.add_number("Camera Toward Y", initial_value=0.0, step=0.1)
        cam_toward_z = server.gui.add_number("Camera Toward Z", initial_value=1.05, step=0.01)
        fov = server.gui.add_number("Camera FOV", initial_value=45.0, step=1.0)
        resolution_width = server.gui.add_number(label="resolution_width", initial_value=1920, step=100)
        resolution_height = server.gui.add_number(label="resolution_height", initial_value=1080, step=100)
        set_cam_button = server.gui.add_button("Set Camera")
        save_img_button = server.gui.add_button("Save Image")
        def set_camera(_=None):
            clients = list(server.get_clients().values())
            client = clients[0] if len(clients) > 0 else None
            if client is not None:
                z = cam_pos_z.value
                toward_z = cam_toward_z.value
                if track_x.value:
                    x = human_motion_dict['transl'][timestep_slider.value, 0] + cam_pos_x.value
                    toward_x = human_motion_dict['transl'][timestep_slider.value, 0] + cam_toward_x.value
                else:
                    x = cam_pos_x.value
                    toward_x = cam_toward_x.value
                if track_y.value:
                    y = human_motion_dict['transl'][timestep_slider.value, 1] + cam_pos_y.value
                    toward_y = human_motion_dict['transl'][timestep_slider.value, 1] + cam_toward_y.value
                else:
                    y = cam_pos_y.value
                    toward_y = cam_toward_y.value
                client.camera.position = np.array([x, y, z])
                client.camera.look_at = np.array([toward_x, toward_y, toward_z])
                client.camera.fov = fov.value * np.pi / 360.0

        for i in [cam_pos_x, cam_pos_y, cam_pos_z, cam_toward_x, cam_toward_y, cam_toward_z, fov]:
            i.on_update(set_camera)
        set_cam_button.on_click(set_camera)

        @save_img_button.on_click
        def _(_):
            clients = list(server.get_clients().values())
            client = clients[0] if len(clients) > 0 else None
            if client is not None:
                img = client.camera.get_render(
                    width=int(resolution_width.value),
                    height=int(resolution_height.value),
                    transport_format='png'
                )
                img_path = f'./save_img_folder/{case_name}_frame{timestep_slider.value}.png'
                img_pil = Image.fromarray(img)
                img_pil.save(img_path)
                print(f"Saved image to {img_path}")

    with server.gui.add_folder('Background Settings'):
        box_size = 100
        ground_grid = server.scene.add_grid(
            "/grid", width=box_size, height=box_size, position=(0.0, -40.0, 0.01),
            cell_size=4, cell_thickness=1, section_size=4,
            plane="xy", shadow_opacity=0.0, 
        )
        bg_box = server.scene.add_box(
            name="/box", dimensions=(box_size, box_size, box_size), position=(0.0, -40.0, -box_size//2),
            color=(170, 170, 150), material='standard', side='double',
            cast_shadow=False, flat_shading=True, receive_shadow=True,
            visible=True,
        )
        box_x = server.gui.add_number(label="Box position X", initial_value=0.0, step=5.0)
        box_y = server.gui.add_number(label="Box position Y", initial_value=-40.0, step=5.0)
        current_z = 0.00
        last_z = current_z
        box_z = server.gui.add_number(label="Box position Z", initial_value=current_z, step=0.01)
        def set_box_pos(_):
            bg_box.position = (box_x.value, box_y.value, bg_box.position[2])
            ground_grid.position = (box_x.value, box_y.value, ground_grid.position[2])
        for i in [box_x, box_y]:
            i.on_update(set_box_pos)

        @box_z.on_update
        def _(_):
            nonlocal last_z, current_z
            current_z = box_z.value
            delta_z = current_z - last_z
            last_z = current_z
            bg_box.position = (bg_box.position[0], bg_box.position[1], bg_box.position[2] + delta_z)
            ground_grid.position = (ground_grid.position[0], ground_grid.position[1], ground_grid.position[2] + delta_z)
            cam_pos_z.value = cam_pos_z.value + delta_z
            cam_toward_z.value = cam_toward_z.value + delta_z

    # Put everything inside a GUI folder (optional)
    with server.gui.add_folder("GMR filter List"):
        # Display the list as a text block
        gui_nrdf_slider = server.gui.add_slider("Nrdf threshold Slider", min=0, max=1, step=0.01, initial_value=0.2)
        gui_nrdf_value_ours = server.gui.add_number("Ours NRDF value", initial_value=ours_res['nrdf_value'][0].item())
        gui_nrdf_value_gmr = server.gui.add_number("GMR NRDF value", initial_value=gmr_res['nrdf_value'][0].item())
        nrdf_filter_list = np.where(gmr_res['nrdf_value'] > gui_nrdf_slider.value)[0].tolist()
        num_text = server.gui.add_text("Filtered index", initial_value=str(nrdf_filter_list))

    @gui_nrdf_slider.on_update
    def _(_event) -> None:
        # Read the slider's value and update the text GUI element
        num_text.value = str(np.where(gmr_res['nrdf_value'] > gui_nrdf_slider.value)[0].tolist())
    
    Max_nrdf_value = server.gui.add_number("GMR Max nrdf value", initial_value=gmr_res['nrdf_value'].max().item())
    server.gui.add_number("Ours Max nrdf value", initial_value=ours_res['nrdf_value'].max().item())

    while True:
        with server.atomic():
            if playing.value:
                timestep_slider.value = (timestep_slider.value + 1) % ours_res['num_timesteps']
                set_camera()
            tstep = timestep_slider.value

            nrdf_frame.wxyz = np.array(ours_res['root_rot'][tstep, [3, 0, 1, 2]])
            # nrdf_frame.wxyz = np.array(Ts_world_root[tstep, 3:])
            nrdf_frame.position = np.array(ours_res['root_pos'][tstep])
            ours_urdf_vis.update_cfg(np.array(ours_res['joints'][tstep]))
            gui_nrdf_value_ours.value = ours_res['nrdf_value'][tstep].item()

            gmr_frame.wxyz = np.array(gmr_res['root_rot'][tstep, [3, 0, 1, 2]])
            # nrdf_frame.wxyz = np.array(Ts_world_root[tstep, 3:])
            gmr_frame.position = np.array(gmr_res['root_pos'][tstep])
            gmr_urdf_vis.update_cfg(np.array(gmr_res['joints'][tstep]))
            gui_nrdf_value_gmr.value = gmr_res['nrdf_value'][tstep].item()

            f = int(tstep)
            body_handle.vertices = human_motion_dict['vertices'][f]  # faces are static

            if playing.value and recording.value:
                clients = list(server.get_clients().values())
                client = clients[0] if len(clients) > 0 else None
                if client is not None:
                    img = client.camera.get_render(
                        width=1920,
                        height=1080,
                        transport_format='png'
                    )
                    img_path = os.path.join(record_frame_dir, f'{timestep_slider.value:03d}.png')
                    img_pil = Image.fromarray(img)
                    img_pil.save(img_path)
                    print(f"Saved image to {img_path}")

        time.sleep(0.1)

if __name__ == "__main__":
    main()