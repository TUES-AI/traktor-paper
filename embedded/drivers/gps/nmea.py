"""Minimal NMEA helpers for extracting latitude/longitude from GPS sentences."""


def checksum_ok(sentence):
    if not sentence.startswith('$') or '*' not in sentence:
        return False
    body, checksum = sentence[1:].split('*', 1)
    checksum = checksum[:2]
    if len(checksum) != 2:
        return False

    calc = 0
    for ch in body:
        calc ^= ord(ch)

    try:
        sent = int(checksum, 16)
    except ValueError:
        return False
    return calc == sent


def dm_to_decimal(value, hemi):
    """Convert NMEA degrees/minutes plus hemisphere into signed decimal degrees."""
    if not value or not hemi or '.' not in value:
        return None

    dot = value.find('.')
    deg_len = dot - 2
    if deg_len <= 0:
        return None

    try:
        degrees = float(value[:deg_len])
        minutes = float(value[deg_len:])
    except ValueError:
        return None

    decimal = degrees + (minutes / 60.0)
    if hemi in ('S', 'W'):
        decimal = -decimal
    return decimal


def parse_lat_lon(sentence):
    """Return `(lat, lon)` from valid RMC/GGA sentences, otherwise `None`."""
    if not checksum_ok(sentence):
        return None

    parts = sentence.split(',')
    msg = parts[0]

    if msg.endswith('RMC'):
        if len(parts) < 7 or parts[2] != 'A':
            return None
        lat = dm_to_decimal(parts[3], parts[4])
        lon = dm_to_decimal(parts[5], parts[6])
        return None if lat is None or lon is None else (lat, lon)

    if msg.endswith('GGA'):
        if len(parts) < 7 or parts[6] in ('', '0'):
            return None
        lat = dm_to_decimal(parts[2], parts[3])
        lon = dm_to_decimal(parts[4], parts[5])
        return None if lat is None or lon is None else (lat, lon)

    return None


def extract_sentences(line):
    return ['$' + item.strip() for item in line.split('$') if item.strip()]
