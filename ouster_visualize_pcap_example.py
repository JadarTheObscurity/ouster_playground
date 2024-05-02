import open3d as o3d
from contextlib import closing
import numpy as np
from ouster import client, pcap

pcap_path = r".\__local\download_2024-04-25_10-16-42\20240420_2309_OS-1-64_991935000698.pcap"
metadata_path = r".\__local\download_2024-04-25_10-16-42\20240420_2309_OS-1-64_991935000698.json"


print("=== Usage ===")
print("Press k: next scan")
print("Press j: previous scan")

# Read ouster metadata
with open(metadata_path, "r") as f:
    info = client.SensorInfo(f.read())

# Get ouster pcap data
source = pcap.Pcap(pcap_path, info)

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()
cloud = o3d.geometry.PointCloud()
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(1.0)  # type: ignore

scan_idx = 0
scan_history = []

def load_scan():
    global scan_idx
    scan = scan_history[scan_idx] 
    xyz = client.XYZLut(info)(scan).reshape(-1, 3)
    cloud.points = o3d.utility.Vector3dVector(xyz)

with closing(client.Scans(source)) as scans:
    scans_iter = iter(scans)
    scan_history.append(next(scans_iter)) # load first scan
    load_scan()

    vis.add_geometry(cloud)
    vis.add_geometry(axes)

    def next_scan(vis):
        global scan_idx
        scan_idx += 1
        if scan_idx >= len(scan_history):
            scan_history.append(next(scans_iter)) # load first scan
        load_scan()
        vis.update_geometry(cloud)
    
    def prev_scan(vis):
        global scan_idx
        scan_idx = scan_idx - 1 if scan_idx > 0 else 0
        load_scan()
        vis.update_geometry(cloud)

    
    vis.register_key_callback(74, prev_scan)
    vis.register_key_callback(75, next_scan)
    vis.run()