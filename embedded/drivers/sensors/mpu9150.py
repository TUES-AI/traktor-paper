"""Minimal MPU-9150 9-axis reader using smbus2."""

import math
import time

import smbus2


class MPU9150:
    ADDRESS_DEFAULT = 0x68
    REG_PWR_MGMT_1 = 0x6B
    REG_GYRO_CONFIG = 0x1B
    REG_ACCEL_CONFIG = 0x1C
    REG_ACCEL_XOUT_H = 0x3B
    REG_TEMP_OUT_H = 0x41
    REG_GYRO_XOUT_H = 0x43
    REG_INT_PIN_CFG = 0x37

    MAG_ADDRESS = 0x0C
    MAG_REG_ST1 = 0x02
    MAG_REG_HXL = 0x03
    MAG_REG_CNTL = 0x0A
    MAG_MODE_SINGLE = 0x01

    GYRO_SCALE = {0: 131.0, 1: 65.5, 2: 32.8, 3: 16.4}
    ACCEL_SCALE = {0: 16384.0, 1: 8192.0, 2: 4096.0, 3: 2048.0}

    def __init__(self, bus=1, address=ADDRESS_DEFAULT, gyro_range=0, accel_range=0):
        self._bus = smbus2.SMBus(bus)
        self._addr = address
        self._gyro_range = gyro_range
        self._accel_range = accel_range
        self._mag_available = False
        self._init_device()

    def _init_device(self):
        self._bus.write_byte_data(self._addr, self.REG_PWR_MGMT_1, 0x01)
        time.sleep(0.1)
        self._bus.write_byte_data(self._addr, self.REG_GYRO_CONFIG, self._gyro_range << 3)
        self._bus.write_byte_data(self._addr, self.REG_ACCEL_CONFIG, self._accel_range << 3)
        self._bus.write_byte_data(self._addr, self.REG_INT_PIN_CFG, 0x02)
        time.sleep(0.05)
        try:
            self._bus.read_byte_data(self.MAG_ADDRESS, self.MAG_REG_ST1)
            self._mag_available = True
        except OSError:
            self._mag_available = False

    def _read_word_signed(self, addr, reg):
        high = self._bus.read_byte_data(addr, reg)
        low = self._bus.read_byte_data(addr, reg + 1)
        value = (high << 8) | low
        return value - 65536 if value >= 32768 else value

    def _read_block(self, addr, reg, length):
        return self._bus.read_i2c_block_data(addr, reg, length)

    @staticmethod
    def _signed_16(low, high):
        value = (high << 8) | low
        return value - 65536 if value >= 32768 else value

    def read_accel(self):
        scale = self.ACCEL_SCALE[self._accel_range]
        data = self._read_block(self._addr, self.REG_ACCEL_XOUT_H, 6)
        return {
            'x': self._signed_16(data[1], data[0]) / scale,
            'y': self._signed_16(data[3], data[2]) / scale,
            'z': self._signed_16(data[5], data[4]) / scale,
        }

    def read_gyro(self):
        scale = self.GYRO_SCALE[self._gyro_range]
        data = self._read_block(self._addr, self.REG_GYRO_XOUT_H, 6)
        return {
            'x': self._signed_16(data[1], data[0]) / scale,
            'y': self._signed_16(data[3], data[2]) / scale,
            'z': self._signed_16(data[5], data[4]) / scale,
        }

    def read_temp_c(self):
        raw = self._read_word_signed(self._addr, self.REG_TEMP_OUT_H)
        return raw / 340.0 + 36.53

    def read_mag(self):
        if not self._mag_available:
            return None
        self._bus.write_byte_data(self.MAG_ADDRESS, self.MAG_REG_CNTL, self.MAG_MODE_SINGLE)
        time.sleep(0.01)
        try:
            status = self._bus.read_byte_data(self.MAG_ADDRESS, self.MAG_REG_ST1)
            if not status & 0x01:
                return None
            data = self._read_block(self.MAG_ADDRESS, self.MAG_REG_HXL, 6)
        except OSError:
            return None
        scale = 0.3
        return {
            'x': self._signed_16(data[0], data[1]) * scale,
            'y': self._signed_16(data[2], data[3]) * scale,
            'z': self._signed_16(data[4], data[5]) * scale,
        }

    def read_orientation(self, accel=None):
        accel = accel or self.read_accel()
        ax, ay, az = accel['x'], accel['y'], accel['z']
        roll = math.degrees(math.atan2(ay, az))
        pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
        return {'roll': roll, 'pitch': pitch}

    def read_all(self):
        accel = self.read_accel()
        return {
            'accel': accel,
            'gyro': self.read_gyro(),
            'temp_c': self.read_temp_c(),
            'orientation': self.read_orientation(accel),
            'mag': self.read_mag(),
        }

    def close(self):
        self._bus.close()
