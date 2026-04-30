#!/usr/bin/env python3
"""TCP camera server for Raspberry Pi rover telemetry.

Protocol per frame:
- 4 bytes big-endian metadata JSON length
- metadata JSON bytes: `{L, F, R}` ultrasonic distances in centimeters
- 4 bytes big-endian JPEG length
- JPEG frame bytes
"""

import json
import socket
import struct
import time

import cv2
import numpy as np

import _paths  # noqa: F401
from api import RoverAPI

HOST = '0.0.0.0'
PORT = 5000
JPEG_QUALITY = 70
FRAME_SIZE = (640, 480)
TARGET_FPS = 15


def _distance_or_max(value, max_distance_cm=300.0):
    return value if value is not None else max_distance_cm


def _stream(rover, conn):
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    while True:
        started = time.monotonic()
        raw = rover.get_ultrasonic()
        metadata = {
            'L': round(_distance_or_max(raw.get(1)), 1),
            'F': round(_distance_or_max(raw.get(3)), 1),
            'R': round(_distance_or_max(raw.get(2)), 1),
        }

        frame_rgb = rover.getframe()
        frame_bgr = cv2.cvtColor(np.array(frame_rgb), cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.resize(frame_bgr, FRAME_SIZE)
        ok, jpeg = cv2.imencode('.jpg', frame_bgr, encode_params)
        if not ok:
            continue

        meta_bytes = json.dumps(metadata).encode('utf-8')
        jpeg_bytes = jpeg.tobytes()
        conn.sendall(struct.pack('>I', len(meta_bytes)))
        conn.sendall(meta_bytes)
        conn.sendall(struct.pack('>I', len(jpeg_bytes)))
        conn.sendall(jpeg_bytes)

        elapsed = time.monotonic() - started
        time.sleep(max(0.0, (1.0 / TARGET_FPS) - elapsed))


def main():
    rover = RoverAPI()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f'Camera server listening on {HOST}:{PORT}')

    try:
        while True:
            conn, addr = server.accept()
            print(f'Viewer connected from {addr}')
            try:
                _stream(rover, conn)
            except (BrokenPipeError, ConnectionResetError):
                print('Viewer disconnected.')
            finally:
                conn.close()
    finally:
        rover.close()
        server.close()


if __name__ == '__main__':
    raise SystemExit(main())
