"""Raw UART reader configured for GPS modules on Raspberry Pi."""

import os
import termios

BAUD_RATES = {
    1200: termios.B1200,
    2400: termios.B2400,
    4800: termios.B4800,
    9600: termios.B9600,
    19200: termios.B19200,
    38400: termios.B38400,
    57600: termios.B57600,
    115200: termios.B115200,
}


def configure_raw_uart(fd, baud):
    if baud not in BAUD_RATES:
        raise ValueError(f'Unsupported baud rate: {baud}')

    attrs = termios.tcgetattr(fd)
    attrs[0] &= ~(
        termios.IGNBRK
        | termios.BRKINT
        | termios.PARMRK
        | termios.ISTRIP
        | termios.INLCR
        | termios.IGNCR
        | termios.ICRNL
        | termios.IXON
        | termios.IXOFF
        | termios.IXANY
    )
    attrs[1] &= ~termios.OPOST
    attrs[2] &= ~(termios.CSIZE | termios.PARENB | termios.CSTOPB)
    attrs[2] |= termios.CS8 | termios.CREAD | termios.CLOCAL
    if hasattr(termios, 'CRTSCTS'):
        attrs[2] &= ~termios.CRTSCTS
    attrs[3] &= ~(termios.ECHO | termios.ECHONL | termios.ICANON | termios.ISIG | termios.IEXTEN)
    attrs[4] = BAUD_RATES[baud]
    attrs[5] = BAUD_RATES[baud]
    attrs[6][termios.VMIN] = 1
    attrs[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, attrs)


class UartReader:
    def __init__(self, port='/dev/ttyAMA0', baud=9600):
        self.port = port
        self.baud = baud
        self.fd = None

    def open(self):
        self.fd = os.open(self.port, os.O_RDONLY | os.O_NOCTTY)
        configure_raw_uart(self.fd, self.baud)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def read(self, size=1024):
        if self.fd is None:
            raise RuntimeError('UART is not open')
        return os.read(self.fd, size)

    def iter_lines(self):
        if self.fd is None:
            raise RuntimeError('UART is not open')

        buffer = ''
        while True:
            chunk = self.read(1024)
            if not chunk:
                continue
            buffer += chunk.decode('ascii', errors='ignore')
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()
                if line:
                    yield line
