#!/usr/bin/env python3
"""Print raw bytes from the GPS UART, similar to a simple minicom session."""

import sys

import _paths  # noqa: F401
from drivers.gps.uart import UartReader
from hardware_pins import GPS_BAUD, GPS_PORT


def main():
    reader = UartReader(port=GPS_PORT, baud=GPS_BAUD)
    try:
        reader.open()
    except OSError as exc:
        print(f'Failed to open {GPS_PORT}: {exc}', file=sys.stderr)
        return 1

    print(f'Reading raw UART from {GPS_PORT} at {GPS_BAUD} baud. Ctrl+C to stop.', file=sys.stderr)
    try:
        while True:
            data = reader.read(1024)
            if data:
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
    except KeyboardInterrupt:
        print('Stopped.', file=sys.stderr)
        return 0
    finally:
        reader.close()


if __name__ == '__main__':
    raise SystemExit(main())
