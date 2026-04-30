#!/usr/bin/env python3
"""Print latitude/longitude fixes from GPS UART, with optional fallback coords."""

import sys
import time

import _paths  # noqa: F401
from drivers.gps.nmea import extract_sentences, parse_lat_lon
from drivers.gps.provider import GPSProvider
from drivers.gps.uart import UartReader
from hardware_pins import GPS_BAUD, GPS_FALLBACK_FILE, GPS_PORT

STATUS_INTERVAL_SECONDS = 5.0


def main():
    gps_provider = GPSProvider(port=GPS_PORT, baud=GPS_BAUD, fallback_file=GPS_FALLBACK_FILE)
    fallback = gps_provider._load_fallback_coords()
    reader = UartReader(port=GPS_PORT, baud=GPS_BAUD)

    try:
        reader.open()
    except OSError as exc:
        print(f'Failed to open {GPS_PORT}: {exc}', file=sys.stderr)
        if fallback is None:
            print('NO_FIX')
        else:
            print(f'{fallback[0]:.6f},{fallback[1]:.6f} (FALLBACK)')
        return 1

    print(f'Reading GPS on {GPS_PORT} at {GPS_BAUD} baud. Waiting for fix...', file=sys.stderr)
    last_status = 0.0

    try:
        for line in reader.iter_lines():
            got_fix = False
            for sentence in extract_sentences(line):
                latlon = parse_lat_lon(sentence)
                if latlon is None:
                    continue
                print(f'{latlon[0]:.6f},{latlon[1]:.6f}')
                sys.stdout.flush()
                got_fix = True

            now = time.monotonic()
            if not got_fix and now - last_status >= STATUS_INTERVAL_SECONDS:
                if fallback is None:
                    print('NO_FIX')
                else:
                    print(f'{fallback[0]:.6f},{fallback[1]:.6f} (FALLBACK)')
                sys.stdout.flush()
                last_status = now
    except KeyboardInterrupt:
        print('Stopped.', file=sys.stderr)
        return 0
    finally:
        reader.close()


if __name__ == '__main__':
    raise SystemExit(main())
