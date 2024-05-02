import sys
sys.path.append(r"D:\1. Project\RaceCar\Exercise\LiDAR_ground_removal\module")

import open3d as o3d
from contextlib import closing
from more_itertools import nth
import numpy as np
from ouster import client, pcap
from ground_removal import Processor
from ouster.mapping.slam import KissBackend
from tqdm import tqdm

pcap_path = r".\__local\download_2024-04-25_10-16-42\20240420_2309_OS-1-64_991935000698.pcap"
metadata_path = r".\__local\download_2024-04-25_10-16-42\20240420_2309_OS-1-64_991935000698.json"
# pcap_path = r"D:\1. Project\RaceCar\Exercise\Ouster\__local\20240420_2310_OS-1-64_991935000698.pcap"
# metadata_path = r"D:\1. Project\RaceCar\Exercise\Ouster\__local\20240420_2310_OS-1-64_991935000698.json"

# create a directory at __local/visualization/{date:time} to store the visualization images
# use date in the directory
import os
import datetime
import time
date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# vis_dir = f"__local/ouster_vis/{date_time}"
# os.makedirs(vis_dir, exist_ok=True)


with open(metadata_path, "r") as f:
    info = client.SensorInfo(f.read())

source = pcap.Pcap(pcap_path, info)


# Hyperparameters
n_segments = 120
n_bins = 30
r_min = 0.3
r_max = 30
line_search_angle = 1
max_dist_to_line = 0.05
sensor_height = 0
max_start_height = 1
long_threshold = 8
segment_step = 2 * np.pi / n_segments
bin_step = (r_max - r_min) / n_bins

processor = Processor(
    n_segments=n_segments,
    n_bins=n_bins,
    r_min=r_min,
    r_max=r_max,
    line_search_angle=line_search_angle,
    max_dist_to_line=max_dist_to_line,
    sensor_height=sensor_height,
    max_start_height=max_start_height,
    long_threshold=long_threshold
)

class Cone:
    def __init__(self, center):
        self.center = center
        self.seen = 1


vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

vis.set_view_status("""
{
    "class_name" : "ViewTrajectory",
    "interval" : 29,
    "is_loop" : false,
    "trajectory" : 
    [
        {
            "boundingbox_max" : [ 13.866294989839913, 15.316081794128472, 1.430556789044076 ],
            "boundingbox_min" : [ 0.0, 0.0, -0.52194880547252298 ],
            "field_of_view" : 60.0,
            "front" : [ -0.87725295095985789, 0.21493793424063554, 0.42921899358787624 ],
            "lookat" : [ 6.9331474949199565, 7.658040897064236, 0.45430399178577652 ],
            "up" : [ 0.36986081818135919, -0.26732030574103149, 0.88979931968547676 ],
            "zoom" : 1.1279999999999999

        }
    ],
    "version_major" : 1,
    "version_minor" : 0
}
                    """)

def _connect_line(i):
    idx = i * 8
    line = [[idx + 0, idx + 1], [idx + 2, idx + 3], 
            [idx + 4, idx + 5], [idx + 6, idx + 7], 
            [idx + 0, idx + 4], [idx + 1, idx + 5], [idx + 2, idx + 6], [idx + 3, idx + 7],
            [idx + 0, idx + 2], [idx + 1, idx + 3], [idx + 4, idx + 6], [idx + 5, idx + 7]]
    return line

def fliter_cluster(xl, yl, zl):
    x_th = 0.4
    y_th = 0.4
    z_th = 0.5
    return xl < x_th and yl < y_th and zl < z_th

def get_cluster(pcd):
    vis_point_cloud = o3d.geometry.PointCloud()
    vis_point_cloud.points = o3d.utility.Vector3dVector(pcd)
    cluster = vis_point_cloud.cluster_dbscan(eps=0.5, min_points=2, print_progress=True)
    max_cluster_label = np.max(cluster)
    cones = []
    print(f"Cluster num: {max_cluster_label}")
    if max_cluster_label > -1:
        cluster_labels = np.arange(0, max_cluster_label + 1)
        for cluster_label in cluster_labels:
            cluster_points = pcd[cluster == cluster_label]
            min_coord = np.min(cluster_points, axis=0)
            max_coord = np.max(cluster_points, axis=0)
            center = ((max_coord + min_coord) / 2).reshape(-1, 3)
            xl , yl, zl = max_coord - min_coord
            if not fliter_cluster(xl, yl, zl):
                continue
            cones.append(Cone(center.flatten()))
    return cones

def cone_geometry(cones, cone_line_set, color=(1, 0.5, 0.5), filter=True):
    # cones: (N x 3)
    all_corners = np.empty((0, 3))
    lines = []
    valid_cones = cones
    if filter:
        valid_cones = [cone for cone in cones if cone.seen >= 2]
    for idx, cone in enumerate(valid_cones):
        center = cone.center
        xl , yl, zl = 0.25, 0.25, 0.4
        corners_x = np.array([xl, xl, xl, xl, -xl, -xl, -xl, -xl]) / 2 
        corners_y = np.array([yl, yl, -yl, -yl, yl, yl, -yl, -yl]) / 2 
        corners_z = np.array([zl, -zl, zl, -zl, zl, -zl, zl, -zl]) / 2 
        corners = np.row_stack((corners_x, corners_y, corners_z)).T + center

        all_corners = np.concatenate((all_corners, corners), axis=0)
        lines += _connect_line(idx)
        colors_bbox = [color for _ in range(len(lines))]
        cone_line_set.points = o3d.utility.Vector3dVector(all_corners)
        cone_line_set.lines = o3d.utility.Vector2iVector(lines)
        cone_line_set.colors = o3d.utility.Vector3dVector(colors_bbox)

def cone_filter(pose, heading, cones, new_cones):
    # Find the cone that is in line of sight, which is within a radius r and angle a
    # from the heading direction
    r = 4  # radius of line of sight
    a = 50  # angle of line of sight

    for cone in cones:
        cone_center = cone.center
        cone_to_pose_vector = cone_center - pose
        cone_to_pose_distance = np.linalg.norm(cone_to_pose_vector)
        cone_to_pose_vector /= cone_to_pose_distance
        angle_diff = np.arccos(np.dot(heading, cone_to_pose_vector)) * 180 / np.pi
        if cone_to_pose_distance <= r and angle_diff <= a:
            cone.seen -= 1

    for new_cone in new_cones:
        for cone in cones:
            if np.linalg.norm(new_cone.center - cone.center) < 1:
                cone.seen += 2
                break
        else:
            cones.append(new_cone)

    cones = [cone for cone in cones if cone.seen > 0]

vis_cloud = o3d.geometry.PointCloud()
vis_poses = o3d.geometry.LineSet()
poses = np.empty((0, 3))
vis_pose = o3d.geometry.LineSet()
vis_pose.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [0, 0, 1]]))
vis_pose.lines = o3d.utility.Vector2iVector([[0, 1]])
vis_pose.colors = o3d.utility.Vector3dVector([(1, 0, 0)])
tmp_cone_line_set = o3d.geometry.LineSet()
tmp_cone_line_set.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [0, 0, 1]]))
tmp_cone_line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
cone_line_set = o3d.geometry.LineSet()
cone_line_set.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [0, 0, 1]]))
cone_line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
cone_list = []

start_idx = 50
with closing(client.Scans(source)) as scans:
    # Run slam
    slam = KissBackend(scans.metadata, max_range=75, min_range=1, voxel_size=1.0)
    for idx, scan in tqdm(enumerate(scans)):
        if idx < start_idx:
            continue
        # SLAM
        # scan_w_poses.pose is a list where each pose represents a column points' pose.
        # use the first valid scan's column pose as the scan pose
        start_time = time.time()

        scan_w_poses = slam.update(scan)
        step1_time = time.time() - start_time

        col = client.first_valid_column(scan_w_poses)
        scan_pose = scan_w_poses.pose[col]
        rotational_pose = scan_pose[:3, :3]
        translation_pose = scan_pose[:3, 3]

        start_time = time.time()

        # Point Cloud
        ## map point cloud to world frame
        xyz = client.XYZLut(info)(scan).reshape(-1, 3)
        xyz = xyz[xyz[:, 2] < 0]
        xyz = (rotational_pose @ xyz.T).T + translation_pose

        step2_time = time.time() - start_time

        start_time = time.time()

        ## Ground removal
        processor.segments = []
        processor.seg_list = []
        xyz = processor(xyz)

        step3_time = time.time() - start_time

        start_time = time.time()

        ## clustering
        new_cones = get_cluster(xyz)
        cone_filter(translation_pose, rotational_pose[:,0], cone_list, new_cones)

        step4_time = time.time() - start_time

        total_time = step1_time + step2_time + step3_time + step4_time

        step1_percentage = (step1_time / total_time) * 100
        step2_percentage = (step2_time / total_time) * 100
        step3_percentage = (step3_time / total_time) * 100
        step4_percentage = (step4_time / total_time) * 100

        print(f"Slam update time: {step1_time} seconds ({step1_percentage}% of total time)")
        print(f"Pose Calculation time: {step2_time} seconds ({step2_percentage}% of total time)")
        print(f"Ground Removal: {step3_time} seconds ({step3_percentage}% of total time)")
        print(f"Clustering time: {step4_time} seconds ({step4_percentage}% of total time)")
        print(f"Total time: {total_time} seconds")

        # Update geometry
        ## Point cloud
        # show_xyz = client.XYZLut(info)(scan).reshape(-1, 3)
        # show_xyz = (rotational_pose @ show_xyz.T).T + translation_pose
        vis_cloud.points = o3d.utility.Vector3dVector(xyz)
        ## Path
        poses = np.concatenate((poses, translation_pose.reshape(1, 3)), axis=0)
        vis_poses.points = o3d.utility.Vector3dVector(poses)
        vis_poses.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(poses.shape[0] - 1)])
        heading = np.array([[0, 0, 0], rotational_pose[:,0]]) + translation_pose
        vis_pose.points = o3d.utility.Vector3dVector(heading)
        ## Cones
        cone_geometry(cone_list, cone_line_set)
        cone_geometry(new_cones, tmp_cone_line_set, color=(0, 1, 0), filter=False)
        if idx == start_idx:
            vis.add_geometry(vis_cloud, reset_bounding_box=False)
            vis.add_geometry(vis_poses, reset_bounding_box=False)
            vis.add_geometry(cone_line_set, reset_bounding_box=False)
            vis.add_geometry(tmp_cone_line_set, reset_bounding_box=False)
            vis.add_geometry(vis_pose, reset_bounding_box=False)
        else:
            vis.update_geometry(vis_cloud)
            vis.update_geometry(vis_poses)
            vis.update_geometry(cone_line_set)
            vis.update_geometry(tmp_cone_line_set)
            vis.update_geometry(vis_pose)
        vis.poll_events()
        vis.update_renderer()
        # vis.capture_screen_image(f"{vis_dir}/{idx}.png")







    