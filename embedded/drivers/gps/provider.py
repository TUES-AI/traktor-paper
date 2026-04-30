"""GPS position provider built on raw UART and NMEA parsing."""

import os
import select
import time
from pathlib import Path

from .nmea import extract_sentences, parse_lat_lon
from .uart import UartReader


class GPSProvider:
    def __init__(self, port='/dev/ttyAMA0', baud=9600, fallback_file='/home/yasen/gps_fallback.env'):
        self.port = port
        self.baud = baud
        self.fallback_file = fallback_file
        self._reader = UartReader(port=self.port, baud=self.baud)
        self._is_open = False
        self._buffer = ''

    def _load_fallback_coords(self):
        path = Path(self.fallback_file)
        if not path.exists():
            return None

        values = {}
        for raw_line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip()

        try:
            return float(values['GPS_FALLBACK_LAT']), float(values['GPS_FALLBACK_LON'])
        except (KeyError, ValueError):
            return None

    def open(self):
        if not self._is_open:
            self._reader.open()
            self._is_open = True

    def close(self):
        if self._is_open:
            self._reader.close()
            self._is_open = False

    def get_position(self, timeout_seconds=2.0, max_sentences=40, allow_fallback=True):
        """Return `{lat, lon, source, fix}` from GPS, fallback file, or no-fix state."""
        try:
            self.open()
        except OSError:
            return self._fallback_or_empty(allow_fallback)

        deadline = time.monotonic() + timeout_seconds
        seen = 0

        while time.monotonic() < deadline and seen < max_sentences:
            remaining = deadline - time.monotonic()
            ready, _, _ = select.select([self._reader.fd], [], [], max(0.0, remaining))
            if not ready:
                break

            chunk = os.read(self._reader.fd, 1024)
            if not chunk:
                continue

            self._buffer += chunk.decode('ascii', errors='ignore')
            while '\n' in self._buffer:
                raw_line, self._buffer = self._buffer.split('\n', 1)
                for sentence in extract_sentences(raw_line.strip()):
                    seen += 1
                    latlon = parse_lat_lon(sentence)
                    if latlon is not None:
                        lat, lon = latlon
                        return {'lat': lat, 'lon': lon, 'source': 'gps', 'fix': True}
                    if seen >= max_sentences:
                        break

        return self._fallback_or_empty(allow_fallback)

    def _fallback_or_empty(self, allow_fallback):
        if allow_fallback:
            coords = self._load_fallback_coords()
            if coords is not None:
                lat, lon = coords
                return {'lat': lat, 'lon': lon, 'source': 'fallback', 'fix': False}
        return {'lat': None, 'lon': None, 'source': 'none', 'fix': False}
