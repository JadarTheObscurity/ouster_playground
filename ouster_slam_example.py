from contextlib import closing
import numpy as np
from ouster import client, pcap
from ouster.mapping.slam import KissBackend
import open3d as o3d

pcap_path = r".\__local\download_2024-04-25_10-16-42\20240420_2309_OS-1-64_991935000698.pcap"
metadata_path = r".\__local\download_2024-04-25_10-16-42\20240420_2309_OS-1-64_991935000698.json"

with open(metadata_path, "r") as f:
    info = client.SensorInfo(f.read())

# Get ouster pcap data
source = pcap.Pcap(pcap_path, info)
slam = KissBackend(info)
last_scan_pose = np.eye(4)
position = np.zeros(3)


from functools import partial
from ouster.viz import SimpleViz, ScansAccumulator
from ouster.mapping.slam import KissBackend
with closing(client.Scans(source)) as scans:
    slam = KissBackend(scans.metadata, max_range=75, min_range=1, voxel_size=1.0)

    scans_w_poses = map(partial(slam.update), scans)
    scans_acc = ScansAccumulator(info,
                                accum_max_num=10,
                                accum_min_dist_num=1,
                                map_enabled=True,
                                map_select_ratio=0.01)

    viz = SimpleViz(info, scans_accum=scans_acc, rate=0.0)
    viz.run(scans_w_poses)
exit()


with closing(client.Scans(source)) as scans:
    for idx, scan in enumerate(scans):
        scan_w_poses = slam.update(scan)
        continue
        col = client.first_valid_column(scan_w_poses)
        # scan_w_poses.pose is a list where each pose represents a column points' pose.
        # use the first valid scan's column pose as the scan pose
        scan_pose = scan_w_poses.pose[col]
        print(f"idx = {idx} and Scan Pose {scan_pose}")

        # calculate the inverse transformation of the last scan pose
        inverse_last = np.linalg.inv(last_scan_pose)
        # calculate the pose difference by matrix multiplication
        pose_diff = np.dot(inverse_last, scan_pose)
        # extract rotation and translation
        rotation_diff = pose_diff[:3, :3]
        translation_diff = pose_diff[:3, 3]
        position += translation_diff
        print(f"idx = {idx} and position: {position}")
        # print(f"idx = {idx} and Rotation Difference: {rotation_diff}, "
        #     f"Translation Difference: {translation_diff}")
    # visualize scan_w_poses with open3d
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    cloud = o3d.geometry.PointCloud()
    xyz = client.XYZLut(info)(scan_w_poses).reshape(-1, 3)
    cloud.points = o3d.utility.Vector3dVector(xyz)
    vis.add_geometry(cloud)
    vis.run()
    """

