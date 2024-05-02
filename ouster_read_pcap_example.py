from contextlib import closing
import numpy as np
from ouster import client, pcap

pcap_path = r".\__local\download_2024-04-25_10-16-42\20240420_2309_OS-1-64_991935000698.pcap"
metadata_path = r".\__local\download_2024-04-25_10-16-42\20240420_2309_OS-1-64_991935000698.json"

with open(metadata_path, "r") as f:
    info = client.SensorInfo(f.read())

# Get ouster pcap data
source = pcap.Pcap(pcap_path, info)


for packet in source:
    if isinstance(packet, client.LidarPacket):
        # Now we can process the LidarPacket. In this case, we access
        # the measurement ids, timestamps, and ranges
        continue
        measurement_ids = packet.measurement_id
        timestamps = packet.timestamp
        ranges = packet.field(client.ChanField.RANGE)
        print(f'  encoder counts = {measurement_ids.shape}')
        print(f'  timestamps = {timestamps.shape}')
        print(f'  ranges = {ranges.shape}')

    elif isinstance(packet, client.ImuPacket):
        # and access ImuPacket content
        print(f'  acc read time = {packet.accel_ts}') # a_x, a_y, a_z in G
        print(f'  acceleration = {packet.accel}') # a_x, a_y, a_z in G
        print(f'  angular_velocity = {packet.angular_vel}') # av_x, av_y, av_z in deg/sec