#!/usr/bin/env python3
"""Receive CSI1 datagrams; optionally emit full packet data as JSON lines."""

import argparse
import json
import socket
import struct
import sys
import time
from dataclasses import asdict, dataclass

HEADER = struct.Struct("!4sIbBBBHH")


@dataclass(frozen=True)
class Packet:
    sequence: int
    rssi: int
    first_word_invalid: bool
    channel: int
    raw_length: int
    iq: tuple[int, ...]


def decode_packet(data: bytes) -> Packet:
    if len(data) < HEADER.size:
        raise ValueError("truncated CSI1 header")
    magic, seq, rssi, invalid, channel, reserved, raw_length, count = HEADER.unpack_from(data)
    if magic != b"CSI1" or reserved != 0 or invalid not in (0, 1):
        raise ValueError("invalid CSI1 header")
    if not 0 < raw_length <= 2048 or count != raw_length:
        raise ValueError("invalid CSI array length (expected default CSI_FORCE_LLTF=0)")
    if len(data) != HEADER.size + count * 2:
        raise ValueError("CSI datagram size does not match sample count")
    iq = struct.unpack_from(f"!{count}h", data, HEADER.size)
    return Packet(seq, rssi, bool(invalid), channel, raw_length, iq)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--json", action="store_true", help="emit every full CSI array as JSON")
    args = parser.parse_args()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        sock.bind((args.bind, args.port))
        sock.settimeout(0.5)
        print(f"Listening on {args.bind}:{args.port}", file=sys.stderr)
        total = interval_count = invalid = sequence_gaps = 0
        previous = None
        last_report = time.monotonic()
        try:
            while True:
                try:
                    data, _ = sock.recvfrom(65535)
                except socket.timeout:
                    data = None
                if data is not None:
                    try:
                        packet = decode_packet(data)
                    except ValueError as error:
                        invalid += 1
                        print(f"Rejected datagram: {error}", file=sys.stderr)
                    else:
                        if previous is not None:
                            step = (packet.sequence - previous) & 0xFFFFFFFF
                            # Ignore duplicates, reordered packets, and transmitter resets.
                            if 1 < step < 0x80000000:
                                sequence_gaps += step - 1
                        previous = packet.sequence
                        total += 1
                        interval_count += 1
                        if args.json:
                            print(json.dumps(asdict(packet), separators=(",", ":")))
                now = time.monotonic()
                if now - last_report >= 5:
                    print(f"received={total} rate={interval_count / (now - last_report):.1f}/s "
                          f"sequence_gaps={sequence_gaps} invalid={invalid}", file=sys.stderr)
                    last_report = now
                    interval_count = 0
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
