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

pcap_path = r"D:\1. Project\RaceCar\Exercise\Ouster\__local\20240420_2310_OS-1-64_991935000698.pcap"
metadata_path = r"D:\1. Project\RaceCar\Exercise\Ouster\__local\20240420_2310_OS-1-64_991935000698.json"

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
    




vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()
vis_cloud = o3d.geometry.PointCloud()
with closing(client.Scans(source)) as scans:
    # Run slam
    slam = KissBackend(scans.metadata, max_range=75, min_range=1, voxel_size=1.0)
    for idx, scan in tqdm(enumerate(scans)):
        xyz = client.XYZLut(info)(scan).reshape(-1, 3)
        xyz = processor(xyz)
        vis_cloud.points = o3d.utility.Vector3dVector(xyz)
        if idx == 0:
            vis.add_geometry(vis_cloud)
        else:
            vis.update_geometry(vis_cloud)
        vis.poll_events()
        vis.update_renderer()