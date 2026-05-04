#!/usr/bin/env python3
"""Persistent JSON-lines TCP rover control server."""

import argparse
import json
import os
import socket
import threading
import time

import _paths  # noqa: F401
from api.rover_api import RoverAPI
from control.safety import SafetyConfig, SafetyController
from control.two_vector_executor import TwoVectorExecutor, TwoVectorGuide
from drivers.sensors.mpu9150 import MPU9150


class RoverTCPServer:
    def __init__(self, host, port, telemetry_hz):
        self.host = host
        self.port = port
        self.telemetry_hz = telemetry_hz
        self.rover = RoverAPI(camera_enabled=False)
        self.imu = MPU9150(bus=1, address=0x68)
        self.safety = SafetyController(self.rover, imu=self.imu, config=SafetyConfig())
        self.running = True
        self.executing = False
        self.conn = None
        self.conn_file = None
        self.conn_lock = threading.Lock()
        self.manual_until = 0.0
        self.manual_lock = threading.Lock()

    def send(self, obj):
        line = json.dumps(obj, sort_keys=True) + '\n'
        with self.conn_lock:
            if self.conn_file is None:
                return
            try:
                self.conn_file.write(line.encode('utf-8'))
                self.conn_file.flush()
            except OSError:
                self.drop_client()

    def drop_client(self):
        try:
            if self.conn_file is not None:
                self.conn_file.close()
        except OSError:
            pass
        try:
            if self.conn is not None:
                self.conn.close()
        except OSError:
            pass
        self.conn = None
        self.conn_file = None

    def telemetry_loop(self):
        delay = 1.0 / self.telemetry_hz
        while self.running:
            if self.conn_file is not None and not self.executing:
                try:
                    distances = self.safety.read_distances()
                    self.send({'type': 'telemetry', 't': time.time(), **distances})
                except Exception as exc:
                    self.send({'type': 'error', 'where': 'telemetry', 'error': repr(exc)})
            time.sleep(delay)

    def manual_watchdog_loop(self):
        while self.running:
            with self.manual_lock:
                if self.manual_until and time.monotonic() > self.manual_until and not self.executing:
                    self.rover.stop_motors()
                    self.manual_until = 0.0
            time.sleep(0.05)

    def handle_drive(self, msg):
        if self.executing:
            self.send({'type': 'warn', 'message': 'manual ignored while executing'})
            return
        left = msg.get('left', 'stop')
        right = msg.get('right', 'stop')
        speed = float(msg.get('speed', 100.0))
        duration = float(msg.get('duration', 0.45))
        with self.manual_lock:
            if left == 'stop' and right == 'stop':
                self.rover.stop_motors()
                self.manual_until = 0.0
            else:
                self.rover.drive(left, right, left_speed=speed, right_speed=speed)
                self.manual_until = time.monotonic() + duration
        self.send({'type': 'drive_ack', 'left': left, 'right': right, 'speed': speed})

    def execute_worker(self, msg):
        self.executing = True
        self.rover.stop_motors()
        try:
            guide = TwoVectorGuide(float(msg['theta1']), float(msg['d1']), float(msg['theta2']), float(msg['d2']))
            log_dir = msg.get('log_dir', 'data/executor_logs')
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f'tcp_two_vector_{time.strftime("%Y%m%d_%H%M%S")}.csv')
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

    def handle_message(self, msg):
        kind = msg.get('type')
        if kind == 'drive':
            self.handle_drive(msg)
        elif kind == 'execute':
            if self.executing:
                self.send({'type': 'warn', 'message': 'already executing'})
            else:
                threading.Thread(target=self.execute_worker, args=(dict(msg),), daemon=True).start()
        elif kind == 'stop':
            self.rover.stop_motors()
            self.send({'type': 'stop_ack'})
        elif kind == 'ping':
            self.send({'type': 'pong', 't': time.time()})
        else:
            self.send({'type': 'warn', 'message': f'unknown type: {kind}'})

    def serve_client(self, conn, addr):
        conn.settimeout(1.0)
        with self.conn_lock:
            self.drop_client()
            self.conn = conn
            self.conn_file = conn.makefile('rwb', buffering=0)
        self.send({'type': 'hello', 'message': 'rover tcp connected'})
        print(json.dumps({'type': 'client_connected', 'addr': addr[0]}), flush=True)
        try:
            while self.running:
                try:
                    line = self.conn_file.readline()
                except socket.timeout:
                    continue
                if not line:
                    break
                try:
                    self.handle_message(json.loads(line.decode('utf-8')))
                except Exception as exc:
                    self.send({'type': 'error', 'where': 'message', 'error': repr(exc)})
        finally:
            self.rover.stop_motors()
            with self.conn_lock:
                self.drop_client()
            print(json.dumps({'type': 'client_disconnected', 'addr': addr[0]}), flush=True)

    def serve_forever(self):
        bias = self.safety.calibrate_gyro()
        print(json.dumps({'type': 'server_ready', 'protocol': 'tcp_jsonl', 'gyro_z_bias': bias, 'host': self.host, 'port': self.port}), flush=True)
        threading.Thread(target=self.telemetry_loop, daemon=True).start()
        threading.Thread(target=self.manual_watchdog_loop, daemon=True).start()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.port))
            sock.listen(1)
            while self.running:
                conn, addr = sock.accept()
                self.serve_client(conn, addr)

    def close(self):
        self.running = False
        self.rover.stop_motors()
        self.imu.close()
        self.rover.close()


def main():
    parser = argparse.ArgumentParser(description='Run persistent JSON-lines TCP rover control server.')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--telemetry-hz', type=float, default=5.0)
    args = parser.parse_args()
    server = RoverTCPServer(args.host, args.port, args.telemetry_hz)
    try:
        server.serve_forever()
    finally:
        server.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
