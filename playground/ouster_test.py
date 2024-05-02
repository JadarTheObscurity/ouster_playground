import sys
sys.path.append(r"D:\1. Project\RaceCar\Exercise\LiDAR_ground_removal\module")

import open3d as o3d
from contextlib import closing
from more_itertools import nth
import numpy as np
from ouster import client, pcap
from ground_removal import Processor
from copy import copy

pcap_path = r"D:\1. Project\RaceCar\Exercise\Ouster\__local\download_2024-04-25_10-16-42\20240420_2309_OS-1-64_991935000698.pcap"
metadata_path = r"D:\1. Project\RaceCar\Exercise\Ouster\__local\download_2024-04-25_10-16-42\20240420_2309_OS-1-64_991935000698.json"

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

def _get_cluster(pcd):
    pcd_2d = pcd.copy()
    pcd_2d[:, 2] = 0
    vis_point_cloud = o3d.geometry.PointCloud()
    vis_point_cloud.points = o3d.utility.Vector3dVector(pcd_2d)
    cluster = vis_point_cloud.cluster_dbscan(eps=0.5, min_points=2, print_progress=True)
    max_cluster_label = np.max(cluster)
    print(f"Cluster num: {max_cluster_label}")
    cones = []
    if max_cluster_label > -1:
        cluster_labels = np.arange(0, max_cluster_label + 1)
        for cluster_label in cluster_labels:
            cluster_points = pcd_2d[cluster == cluster_label]
            min_coord = np.min(cluster_points, axis=0)
            max_coord = np.max(cluster_points, axis=0)
            center = ((max_coord + min_coord) / 2).reshape(-1, 3)
            xl , yl, zl = max_coord - min_coord
            if not fliter_cluster(xl, yl, zl):
                continue
            print(center)
            cones.append(center)
    return cones

def get_cluster(pcd, cluster_line_set):
    pcd_2d = pcd.copy()
    # pcd_2d[:, 2] = 0
    vis_point_cloud = o3d.geometry.PointCloud()
    vis_point_cloud.points = o3d.utility.Vector3dVector(pcd_2d)
    cluster = vis_point_cloud.cluster_dbscan(eps=0.5, min_points=2, print_progress=True)
    max_cluster_label = np.max(cluster)
    cones = []
    print(f"Cluster num: {max_cluster_label}")
    if max_cluster_label > -1:
        cluster_labels = np.arange(0, max_cluster_label + 1)
        all_corners = np.empty((0, 3))
        lines = []
        cluster_cnt = 0
        for cluster_label in cluster_labels:
            cluster_points = pcd_2d[cluster == cluster_label]
            min_coord = np.min(cluster_points, axis=0) - 0.02 * np.ones(3)
            max_coord = np.max(cluster_points, axis=0) + 0.02 * np.ones(3)
            center = ((max_coord + min_coord) / 2).reshape(-1, 3)
            xl , yl, zl = max_coord - min_coord
            if not fliter_cluster(xl, yl, zl):
                continue
            cones.append(center.flatten())
            print(center)
            print(f"{cluster_label}, dz: {zl}")
            corners_x = np.array([xl, xl, xl, xl, -xl, -xl, -xl, -xl]) / 2 
            corners_y = np.array([yl, yl, -yl, -yl, yl, yl, -yl, -yl]) / 2 
            corners_z = np.array([zl, -zl, zl, -zl, zl, -zl, zl, -zl]) / 2 
            corners = np.row_stack((corners_x, corners_y, corners_z)).T + center

            all_corners = np.concatenate((all_corners, corners), axis=0)
            lines += _connect_line(cluster_cnt)
            cluster_cnt += 1
        if cluster_cnt != 0: 
            colors_bbox = [(1, 0, 0.5) for _ in range(len(lines))]
            cluster_line_set.points = o3d.utility.Vector3dVector(all_corners)
            cluster_line_set.lines = o3d.utility.Vector2iVector(lines)
            cluster_line_set.colors = o3d.utility.Vector3dVector(colors_bbox)
    return cones

def cone_geometry(cones, cone_line_set):
    # cones: (N x 3)
    all_corners = np.empty((0, 3))
    lines = []
    for idx, center in enumerate(cones):
        xl , yl, zl = 0.25, 0.25, 0.4
        corners_x = np.array([xl, xl, xl, xl, -xl, -xl, -xl, -xl]) / 2 
        corners_y = np.array([yl, yl, -yl, -yl, yl, yl, -yl, -yl]) / 2 
        corners_z = np.array([zl, -zl, zl, -zl, zl, -zl, zl, -zl]) / 2 
        corners = np.row_stack((corners_x, corners_y, corners_z)).T + center

        all_corners = np.concatenate((all_corners, corners), axis=0)
        lines += _connect_line(idx)
        colors_bbox = [(1, 0.5, 0.5) for _ in range(len(lines))]
        cone_line_set.points = o3d.utility.Vector3dVector(all_corners)
        cone_line_set.lines = o3d.utility.Vector2iVector(lines)
        cone_line_set.colors = o3d.utility.Vector3dVector(colors_bbox)

    
def cone_filter(cones, new_cones):
    # if the new cone is far from old cone, add to cones list
    # if the new cone is close to old cone, update the old cone
    # if the new cone is not in the old cone, add to cones list
    for new_cone in new_cones:
        for cone in cones:
            if np.linalg.norm(new_cone - cone) < 0.5:
                cone = new_cone
                break
        else:
            cones.append(new_cone)

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()
cloud = o3d.geometry.PointCloud()
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(1.0)  # type: ignore
cluster_line_set = o3d.geometry.LineSet()
cluster_line_set.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [0, 0, 1]]))
cluster_line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
cone_line_set = o3d.geometry.LineSet()
cone_line_set.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [0, 0, 1]]))
cone_line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
cone_list = []
def update(scans_iter, vis, first_iter=False):
    scan = next(scans_iter)
    processor.segments = []
    processor.seg_list = []
    xyz = client.XYZLut(info)(scan).reshape(-1, 3)
    #xyz = processor(xyz)
    #new_cones = get_cluster(xyz, cluster_line_set)
    #cone_filter(cone_list, new_cones)
    # Visualize
    cloud.points = o3d.utility.Vector3dVector(xyz)
    cone_geometry(cone_list, cone_line_set)

    if first_iter:
        vis.add_geometry(cloud)
        vis.add_geometry(cluster_line_set, reset_bounding_box=False)
        # vis.add_geometry(cone_line_set, reset_bounding_box=False)
    else:
        vis.update_geometry(cloud)
        vis.update_geometry(cluster_line_set)
        # vis.update_geometry(cone_line_set)



start_idx = 150
with closing(client.Scans(source)) as scans:

    scans_iter = iter(scans)
    for i in range(start_idx):
        _ = next(scans_iter)

    update(scans_iter=scans_iter, vis=vis, first_iter=True)

    def next_scan(vis):
        update(scans_iter=scans_iter, vis=vis)
    vis.register_key_callback(75, next_scan)
    
    #create point cloud and coordinate axes geometries
    vis.run()