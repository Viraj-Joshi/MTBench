import os
import math
import numpy as np

from isaacgym import gymutil
from isaacgym import gymapi
from isaacgym.terrain_utils import *

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Jackal Isaac")

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = 0.005
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.shape_collision_margin = 0.25
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 10
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.substeps = 1
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
plane_params.static_friction = 0.5
plane_params.dynamic_friction = 0.5
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# add Jackal robot
asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets_v2')
# asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
# asset_file = "test/wall.xml"
asset_file = "unified_objects/buttonbox.xml"
# asset_file = "scene/basic_scene.xml"
# asset_file = "mjcf/open_ai_assets/hand/egg.xml"
asset_path = os.path.join(asset_root, asset_file)
asset_root = os.path.dirname(asset_path)
asset_file = os.path.basename(asset_path)

asset_options = gymapi.AssetOptions()
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_VEL
asset_options.collapse_fixed_joints = True
asset_options.replace_cylinder_with_capsule = True
# asset_options.flip_visual_attachments = True
# asset_options.density = 0.001
asset_options.angular_damping = 0.0
asset_options.linear_damping = 0.0
asset_options.armature = 0.0
asset_options.thickness = 0.01
asset_options.disable_gravity = False
asset_options.fix_base_link = True

franka_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

spacing = 5.
num_envs = 16
enable_viewer_sync = False

env_lower = gymapi.Vec3(-spacing/2, -spacing, 0)
env_upper = gymapi.Vec3(spacing/2, spacing, spacing)
envs = []

for i in range(num_envs):
    env_handle = gym.create_env(sim, env_lower, env_upper, int(np.sqrt(num_envs)))
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(*[2.25, 2.25, 1.0])
    pose.r = (
        gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), math.pi * 90. / 180.)
    )
    franka_handle = gym.create_actor(env_handle, franka_asset, pose, "franka", i, 0, 0)
    # gym.set_actor_dof_properties(env_handle, franka_handle, dof_props)

cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# step the simulation for 60 (s)
for k in range(12000):
    rigid_body_states = gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL)

    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # step graphics
    if enable_viewer_sync:
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)

    else:
        gym.poll_viewer_events(viewer)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)