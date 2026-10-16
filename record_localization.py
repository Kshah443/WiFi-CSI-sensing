#!/usr/bin/env python3
"""Record one stationary ESP32 CSI localization trial on Ubuntu.

Setup (Ubuntu):
    python3 -m venv .venv
    source .venv/bin/activate
    pip install pyserial

Usage:
    python3 record_localization.py P1 01
    python3 record_localization.py P9 05 --port /dev/ttyUSB0 --baud 921600
    python3 record_localization.py P1 01 --countdown 15 --duration 20
    python3 record_localization.py P1 01 --overwrite
    python3 record_localization.py P1 01 --transport udp --udp-port 5005 \
        --output-dir /home/komail-shah/csi_localization_diagonal/

Defaults: 10 s to move, 3 s stabilization, 20 s capture; output is
~/csi_localization/P1/P1_01.txt. Close other serial monitors first.
If permission is denied, add your Ubuntu user to the dialout group:
    sudo usermod -aG dialout "$USER"
Then log out and back in. Never move the boards between trials.

UDP mode imports the unchanged project decoder from
firmware.csi_recv.tools.receive_udp (run from your project root). Default bind:
0.0.0.0:5005. It saves CSI_DATA,sequence,rssi,first_word_invalid,channel,len,"[I,Q,...]"
with all 234 values; no array cropping, normalization, or invalid-word removal.
Duplicate and out-of-order packets are counted and excluded. Gaps count forward
sequence jumps (late arrivals do not erase observed gaps); uint32 wrap is handled.
A transmitter reset requires restarting the trial. Counters cover capture only.
Only one sender is allowed. UDP has no sender timestamp, so delayed in-flight
pre-capture packets cannot be distinguished after clearing the local queue.

Only newline-terminated lines containing CSI_DATA are retained as raw bytes.
The first post-reset line is discarded conservatively (it may start mid-line).
An unfinished final line is discarded. Timing uses a monotonic clock; Python
and Ubuntu are not hard real-time, so scheduling can slightly delay returning,
but lines observed at or after the deadline are never accepted.
Failed/empty/interrupted recordings do not create or replace a trial file.
"""

import argparse
import math
import socket
import os
from pathlib import Path
import sys
import tempfile
import time

MAX_LINE_BYTES = 65536


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return number


def nonnegative(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def udp_port(value):
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return number


class UDPConnection:
    """Adapter for the same preparation/draining loop as serial."""

    def __init__(self, bind, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
            self.sock.bind((bind, port))
        except BaseException:
            self.sock.close()
            raise
        self.timeout = 0.1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.sock.close()

    @property
    def in_waiting(self):
        return 1

    def read(self, size):
        self.sock.settimeout(self.timeout)
        try:
            return self.sock.recvfrom(65535)[0]
        except socket.timeout:
            return b""

    def reset_input_buffer(self):
        self.sock.setblocking(False)
        # Drain the local queue; bound cleanup under sustained overload.
        import select
        stop = time.monotonic() + 0.25
        while select.select([self.sock], [], [], 0)[0]:
            self.sock.recvfrom(65535)
            if time.monotonic() >= stop:
                raise OSError("UDP input could not be drained; lower sender rate and retry")


def udp_line(packet):
    """Available wire metadata + quoted trailing full I/Q list, no fake fields.

    CSV columns: type, sequence, rssi, first_word_invalid, channel, len, data.
    Compatible with CSI_DATA parsers extracting the trailing quoted I/Q list.
    The UDP protocol does not carry the other ESP-IDF serial CSV metadata.
    """
    values = ",".join(str(value) for value in packet.iq)
    return (f'CSI_DATA,{packet.sequence},{packet.rssi},'
            f'{int(packet.first_word_invalid)},{packet.channel},'
            f'{packet.raw_length},"[{values}]"\n').encode("ascii")


def capture_udp(connection, duration, decode_packet):
    connection.reset_input_buffer()
    deadline = time.monotonic() + duration
    packets = []
    stats = dict(malformed=0, wrong_length=0, sequence_gaps=0,
                 duplicates=0, out_of_order=0)
    seen = set()
    highest = None
    sender = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        connection.sock.settimeout(min(0.05, remaining))
        try:
            data, address = connection.sock.recvfrom(65535)
        except socket.timeout:
            continue
        if time.monotonic() >= deadline:
            break
        try:
            packet = decode_packet(data)
        except (ValueError, OverflowError):
            stats["malformed"] += 1
            continue
        # This experiment requires the complete 234-value (117-complex) array.
        if packet.raw_length != 234 or len(packet.iq) != 234:
            stats["wrong_length"] += 1
            continue
        if sender is None:
            sender = address
        elif address != sender:
            raise OSError("Multiple UDP senders detected; isolate one receiver per trial")
        if packet.sequence in seen:
            stats["duplicates"] += 1
            continue
        seen.add(packet.sequence)
        if highest is not None:
            step = (packet.sequence - highest) & 0xFFFFFFFF
            if step >= 0x80000000:
                stats["out_of_order"] += 1
                continue  # Preserve chronological windows; resets require a new trial.
            stats["sequence_gaps"] += max(0, step - 1)
        highest = packet.sequence
        packets.append(udp_line(packet))
    print("UDP counters: " + " ".join(f"{key}={value}" for key, value in stats.items()),
          flush=True)
    return packets


def arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("position", choices=[f"P{i}" for i in range(1, 10)])
    parser.add_argument("trial", choices=[f"{i:02d}" for i in range(1, 6)])
    parser.add_argument("--transport", choices=("serial", "udp"), default="serial")
    parser.add_argument("--udp-bind", default="0.0.0.0")
    parser.add_argument("--udp-port", type=udp_port, default=5005)
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=positive_int, default=921600)
    parser.add_argument("--countdown", type=nonnegative, default=10.0)
    parser.add_argument("--stabilization", type=nonnegative, default=3.0)
    parser.add_argument("--duration", type=positive, default=20.0)
    parser.add_argument("--output-dir", type=Path, default=Path("~/csi_localization"))
    parser.add_argument("--min-packets", type=positive_int, default=100,
                        help="warn below this count (default: 100; adjust to firmware rate)")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def discard_for(connection, seconds, label):
    """Drain incoming bytes throughout preparation, without retaining them."""
    deadline = time.monotonic() + seconds
    last_display = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        display = math.ceil(remaining)
        if display != last_display:
            print(f"{label}: {display} s", flush=True)
            last_display = display
        connection.timeout = min(0.1, remaining)
        connection.read(min(4096, max(1, connection.in_waiting)))


def capture(connection, duration):
    """Bound reads by the deadline, and assemble complete lines across reads."""
    connection.reset_input_buffer()
    start = time.monotonic()
    deadline = start + duration
    packets = []
    pending = bytearray()
    discard_line = True  # A reset can land in the middle of an on-wire line.
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        connection.timeout = min(0.05, remaining)
        chunk = connection.read(min(4096, max(1, connection.in_waiting)))
        if time.monotonic() >= deadline:
            break
        for byte in chunk:
            if byte == 10:
                if not discard_line and b"CSI_DATA" in pending:
                    packets.append(bytes(pending) + b"\n")
                pending.clear()
                discard_line = False
            elif not discard_line:
                pending.append(byte)
                if len(pending) > MAX_LINE_BYTES:
                    pending.clear()
                    discard_line = True  # Resynchronize after malformed/oversized input.
    return packets


def save_trial(path, packets, overwrite):
    """Publish only a completed nonempty trial; protect existing files atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix=".csi-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.writelines(packets)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Same-directory hard link fails atomically if the target already exists.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    args = arguments()
    path = args.output_dir.expanduser() / args.position / f"{args.position}_{args.trial}.txt"
    if path.exists() and not args.overwrite:
        print(f"Error: {path} already exists. Use --overwrite explicitly to replace it.",
              file=sys.stderr)
        return 1
    errors = (OSError,)
    if args.transport == "serial":
        try:
            import serial
        except ImportError:
            print("Missing PySerial. Install it with: pip install pyserial", file=sys.stderr)
            return 1
        errors += (serial.SerialException,)
    else:
        try:
            from firmware.csi_recv.tools.receive_udp import decode_packet
        except ImportError as error:
            print(f"Cannot import the project UDP decoder: {error}. "
                  "Run from the project root containing firmware/csi_recv/tools/receive_udp.py.",
                  file=sys.stderr)
            return 1
    print(f"Target: {path}", flush=True)
    try:
        if args.transport == "serial":
            connection = serial.Serial(args.port, args.baud, timeout=0.1, exclusive=True)
            description = f"Serial open: {args.port} at {args.baud} baud"
        else:
            connection = UDPConnection(args.udp_bind, args.udp_port)
            description = f"UDP listening: {args.udp_bind}:{args.udp_port}"
        with connection:
            print(f"{description}. Move to {args.position}.", flush=True)
            discard_for(connection, args.countdown, "Reach position")
            print("Hold still for stabilization.", flush=True)
            discard_for(connection, args.stabilization, "Stabilizing")
            print(f"Recording for {args.duration:g} s. Hold still until completion.", flush=True)
            if args.transport == "serial":
                packets = capture(connection, args.duration)
            else:
                packets = capture_udp(connection, args.duration, decode_packet)
        count = len(packets)
        print(f"Captured CSI packets: {count}", flush=True)
        if not packets:
            print("Error: empty recording rejected; no trial file saved.", file=sys.stderr)
            return 1
        if count < args.min_packets:
            print(f"Warning: suspiciously low packet count: {count} "
                  f"({count / args.duration:.2f} packets/s); threshold {args.min_packets}. "
                  "Check transmitter, receiver, and firmware rate before using this trial.",
                  file=sys.stderr)
        save_trial(path, packets, args.overwrite)
        print(f"Done. Saved {path}", flush=True)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Capture rejected; no trial file saved or replaced.", file=sys.stderr)
        return 130
    except errors as error:
        print(f"Error: {error}. Capture was not successfully committed.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
