#!/usr/bin/env python3
"""Persistent WebSocket control server for the rover.

This process owns the rover GPIO resources. The Mac UI should talk to this
server instead of starting separate SSH scripts for telemetry, WASD, and guide
execution.
"""

import argparse
import base64
import hashlib
import json
import os
import socket
import struct
import threading
import time

import _paths  # noqa: F401
from api.rover_api import RoverAPI
from control.safety import SafetyConfig, SafetyController
from control.two_vector_executor import TwoVectorExecutor, TwoVectorGuide
from drivers.sensors.mpu9150 import MPU9150


GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'


def recv_exact(conn, n):
    data = b''
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            raise ConnectionError('socket closed')
        data += chunk
    return data


def recv_ws_text(conn):
    header = recv_exact(conn, 2)
    b1, b2 = header
    opcode = b1 & 0x0F
    masked = bool(b2 & 0x80)
    length = b2 & 0x7F
    if opcode == 0x8:
        raise ConnectionError('websocket closed')
    if length == 126:
        length = struct.unpack('!H', recv_exact(conn, 2))[0]
    elif length == 127:
        length = struct.unpack('!Q', recv_exact(conn, 8))[0]
    mask = recv_exact(conn, 4) if masked else b''
    payload = recv_exact(conn, length) if length else b''
    if masked:
        payload = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
    if opcode == 0x9:
        send_ws_text(conn, '')
        return None
    if opcode != 0x1:
        return None
    return payload.decode('utf-8')


def send_ws_text(conn, text):
    payload = text.encode('utf-8')
    length = len(payload)
    if length < 126:
        header = struct.pack('!BB', 0x81, length)
    elif length < 65536:
        header = struct.pack('!BBH', 0x81, 126, length)
    else:
        header = struct.pack('!BBQ', 0x81, 127, length)
    conn.sendall(header + payload)


def websocket_handshake(conn):
    request = b''
    while b'\r\n\r\n' not in request:
        chunk = conn.recv(4096)
        if not chunk:
            raise ConnectionError('closed during handshake')
        request += chunk
    headers = {}
    lines = request.decode('utf-8', errors='replace').split('\r\n')
    for line in lines[1:]:
        if ':' in line:
            key, value = line.split(':', 1)
            headers[key.strip().lower()] = value.strip()
    key = headers['sec-websocket-key']
    accept = base64.b64encode(hashlib.sha1((key + GUID).encode('ascii')).digest()).decode('ascii')
    response = (
        'HTTP/1.1 101 Switching Protocols\r\n'
        'Upgrade: websocket\r\n'
        'Connection: Upgrade\r\n'
        f'Sec-WebSocket-Accept: {accept}\r\n\r\n'
    )
    conn.sendall(response.encode('ascii'))


class RoverWSServer:
    def __init__(self, host, port, telemetry_hz):
        self.host = host
        self.port = port
        self.telemetry_hz = telemetry_hz
        self.rover = RoverAPI(camera_enabled=False)
        self.imu = MPU9150(bus=1, address=0x68)
        self.safety = SafetyController(self.rover, imu=self.imu, config=SafetyConfig())
        self.conn = None
        self.conn_lock = threading.Lock()
        self.running = True
        self.executing = False
        self.manual_until = 0.0
        self.manual_lock = threading.Lock()

    def send(self, obj):
        line = json.dumps(obj, sort_keys=True)
        with self.conn_lock:
            if self.conn is None:
                return
            try:
                send_ws_text(self.conn, line)
            except OSError:
                self.conn = None

    def telemetry_loop(self):
        delay = 1.0 / self.telemetry_hz
        while self.running:
            if self.conn is not None and not self.executing:
                try:
                    distances = self.safety.read_distances()
                    self.send({'type': 'telemetry', 't': time.time(), **distances})
                except Exception as exc:
                    self.send({'type': 'error', 'where': 'telemetry', 'error': str(exc)})
            time.sleep(delay)

    def manual_watchdog_loop(self):
        while self.running:
            with self.manual_lock:
                expired = self.manual_until and time.monotonic() > self.manual_until
                if expired and not self.executing:
                    self.rover.stop_motors()
                    self.manual_until = 0.0
            time.sleep(0.05)

    def handle_drive(self, msg):
        if self.executing:
            self.send({'type': 'warn', 'message': 'ignoring manual drive while guide is executing'})
            return
        left = msg.get('left', 'stop')
        right = msg.get('right', 'stop')
        speed = float(msg.get('speed', 90.0))
        duration = float(msg.get('duration', 0.35))
        with self.manual_lock:
            if left == 'stop' and right == 'stop':
                self.rover.stop_motors()
                self.manual_until = 0.0
            else:
                self.rover.drive(left, right, left_speed=speed, right_speed=speed)
                self.manual_until = time.monotonic() + duration
        self.send({'type': 'drive_ack', 'left': left, 'right': right, 'speed': speed})

    def handle_execute(self, msg):
        if self.executing:
            self.send({'type': 'warn', 'message': 'guide already executing'})
            return
        threading.Thread(target=self.execute_worker, args=(dict(msg),), daemon=True).start()

    def execute_worker(self, msg):
        self.executing = True
        self.rover.stop_motors()
        try:
            guide = TwoVectorGuide(
                float(msg['theta1']),
                float(msg['d1']),
                float(msg['theta2']),
                float(msg['d2']),
            )
            log_dir = msg.get('log_dir', 'data/executor_logs')
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f'ws_two_vector_{time.strftime("%Y%m%d_%H%M%S")}.csv')
            executor = TwoVectorExecutor(
                self.safety,
                guide,
                log_path,
                speed_pct=float(msg.get('speed', 85.0)),
                turn_speed_pct=float(msg.get('turn_speed', 80.0)),
                cm_per_second=float(msg.get('cm_per_second', 40.0)),
                dt=float(msg.get('dt', 0.05)),
            )
            self.send({'type': 'execute_started', 'guide': guide.__dict__, 'log_path': log_path})
            report = executor.execute()
            report['summary_path'] = executor.write_summary(report)
            self.send({'type': 'execute_done', 'report': report})
        except Exception as exc:
            self.rover.stop_motors()
            self.send({'type': 'error', 'where': 'execute', 'error': repr(exc)})
        finally:
            self.rover.stop_motors()
            self.executing = False

    def handle_message(self, text):
        if not text:
            return
        msg = json.loads(text)
        kind = msg.get('type')
        if kind == 'drive':
            self.handle_drive(msg)
        elif kind == 'execute':
            self.handle_execute(msg)
        elif kind == 'stop':
            self.rover.stop_motors()
            self.send({'type': 'stop_ack'})
        elif kind == 'ping':
            self.send({'type': 'pong', 't': time.time()})
        else:
            self.send({'type': 'warn', 'message': f'unknown message type: {kind}'})

    def serve_forever(self):
        bias = self.safety.calibrate_gyro()
        print(json.dumps({'type': 'server_ready', 'gyro_z_bias': bias, 'host': self.host, 'port': self.port}), flush=True)
        threading.Thread(target=self.telemetry_loop, daemon=True).start()
        threading.Thread(target=self.manual_watchdog_loop, daemon=True).start()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(1)
            while self.running:
                conn, addr = server.accept()
                print(json.dumps({'type': 'client_connected', 'addr': addr[0]}), flush=True)
                try:
                    websocket_handshake(conn)
                    with self.conn_lock:
                        if self.conn is not None:
                            try:
                                self.conn.close()
                            except OSError:
                                pass
                        self.conn = conn
                    self.send({'type': 'hello', 'message': 'rover websocket connected'})
                    while self.running:
                        text = recv_ws_text(conn)
                        self.handle_message(text)
                except Exception as exc:
                    print(json.dumps({'type': 'client_error', 'error': repr(exc)}), flush=True)
                finally:
                    self.rover.stop_motors()
                    with self.conn_lock:
                        if self.conn is conn:
                            self.conn = None
                    try:
                        conn.close()
                    except OSError:
                        pass

    def close(self):
        self.running = False
        self.rover.stop_motors()
        self.imu.close()
        self.rover.close()


def main():
    parser = argparse.ArgumentParser(description='Run persistent rover WebSocket control server.')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--telemetry-hz', type=float, default=6.0)
    args = parser.parse_args()
    server = RoverWSServer(args.host, args.port, args.telemetry_hz)
    try:
        server.serve_forever()
    finally:
        server.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
