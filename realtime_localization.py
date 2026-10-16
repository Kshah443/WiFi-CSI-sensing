#!/usr/bin/env python3
"""Live CSI inference with metadata-selected raw or relative_shape features.
For relative_shape, also place evaluate_normalization.py and
train_localization_relative.py beside this script.

Dependencies: numpy, scikit-learn, joblib, pyserial (serial only), matplotlib (GUI only).
Examples:
  python realtime_localization.py
  python realtime_localization.py --no-gui --log
  python realtime_localization.py --transport udp --no-gui --results /home/komail-shah/csi_localization_relative_results
  python realtime_localization.py --alpha .25 --log predictions.csv
Only load trusted joblib model files. Probabilities are uncalibrated; the weighted
coordinate is an interpolation of grid classes, not validated continuous tracking.
"""
import argparse
import csv
import json
import logging
import socket
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

try:
    import train_localization as training
except ImportError as exc:
    raise SystemExit('Cannot import train_localization.py. Place the original training script beside this file. ' + str(exc))


def normalize_metadata(value):
    """Canonicalize serialization containers without changing ordered sequences."""
    if isinstance(value, np.ndarray):
        return normalize_metadata(value.tolist())
    if isinstance(value, np.generic):
        return normalize_metadata(value.item())
    if isinstance(value, dict):
        return {k: normalize_metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_metadata(v) for v in value]
    return value


def load_verified(results, window_size, debug_metadata=False):
    """Compare inference contracts, not complete training-run metadata objects."""
    metadata = normalize_metadata(json.loads((results / 'model_metadata.json').read_text()))
    bundle = joblib.load(results / 'localization_model.joblib')
    if not isinstance(bundle, dict) or not isinstance(bundle.get('metadata'), dict):
        raise ValueError('Model artifact must contain model and embedded metadata objects')
    embedded = normalize_metadata(bundle['metadata'])
    model = bundle.get('model')
    if not isinstance(model, RandomForestClassifier):
        raise ValueError('Expected trained RandomForestClassifier')
    expected = {
        'preprocessing_version': training.PREPROCESSING_VERSION,
        'window_size_packets': window_size,
        'window_stride_packets': window_size,
        'feature_order': training.FEATURE_STATS,
        'iq_format': 'alternating signed int8 components; hypot of each pair',
        'std_ddof': 0, 'quantile_method': 'linear', 'dtype': 'float32',
        'normalization': 'none', 'subcarrier_mask': 'all',
        'trailing_packets': 'discard',
        'packet_policy': 'reject malformed/incomplete lines and length mismatches; concatenate accepted packets in original order',
        'coordinates_ft': training.COORDINATES,
        'class_order': [str(x) for x in model.classes_],
    }
    variant = metadata.get('preprocessing_variant', 'raw')
    if embedded.get('preprocessing_variant', 'raw') != variant:
        raise ValueError('Embedded and JSON preprocessing variants differ')
    if variant == 'relative_shape':
        try:
            from train_localization_relative import relative_contract
        except ImportError as exc:
            raise ValueError('relative_shape requires train_localization_relative.py and evaluate_normalization.py') from exc
        contract = relative_contract()
        if window_size != contract['window_size_packets']:
            raise ValueError('relative_shape requires exactly 100-packet windows')
        if normalize_metadata(training.COORDINATES) != normalize_metadata(contract['coordinates_ft']):
            raise ValueError('Live display coordinate mapping differs from relative_shape coordinates')
        expected.update(contract)
    elif variant != 'raw':
        raise ValueError(f'Unsupported preprocessing_variant: {variant!r}')
    elif 'preprocessing_variant' in metadata or 'preprocessing_variant' in embedded:
        expected['preprocessing_variant'] = 'raw'
    length = metadata.get('amplitude_length')
    valid_length = type(length) is int and length > 0
    names = expected['feature_names'] if variant == 'relative_shape' else (training.feature_names(length) if valid_length else [])
    if variant == 'raw':
        expected.update(amplitude_length=length if valid_length else '<positive integer required>',
                        raw_iq_length=length * 2 if valid_length else '<invalid amplitude_length>',
                        feature_names=names, feature_count=len(names))
    expected = normalize_metadata(expected)
    missing = '<MISSING>'
    failures = []
    for key, live_value in expected.items():
        artifact_value = embedded.get(key, missing)
        json_value = metadata.get(key, missing)
        if debug_metadata:
            print(f'METADATA {key}: model-artifact={artifact_value!r}; JSON={json_value!r}; live/model={live_value!r}', flush=True)
        if artifact_value != json_value or json_value != live_value:
            failures.append(f'{key}: model-artifact={artifact_value!r}; JSON={json_value!r}; live/model={live_value!r}')
    # These fields are optional in the original training metadata, but if present
    # they must agree with the actual forest and the canonical class/feature lists.
    for key, actual in [('class_labels', training.LABELS), ('n_features_in_', int(model.n_features_in_))]:
        artifact_value, json_value = embedded.get(key, missing), metadata.get(key, missing)
        if debug_metadata:
            print(f'METADATA {key}: model-artifact={artifact_value!r}; JSON={json_value!r}; live/model={actual!r}', flush=True)
        for origin, value in [('model-artifact', artifact_value), ('JSON', json_value)]:
            if value != missing and value != actual:
                failures.append(f'{key}: {origin}={value!r}; live/model={actual!r}')
    classes = expected['class_order']
    if len(classes) != 9 or set(classes) != set(training.LABELS):
        failures.append(f'class_labels: model={classes!r}; required={training.LABELS!r}')
    if model.n_features_in_ != len(names):
        failures.append(f'n_features_in_: model={model.n_features_in_!r}; JSON feature_count={metadata.get("feature_count")!r}; live={len(names)}')
    if hasattr(model, 'feature_names_in_') and normalize_metadata(model.feature_names_in_) != names:
        failures.append(f'feature_names_in_: model={model.feature_names_in_!r}; JSON/live={names!r}')
    if failures:
        raise ValueError('PREDICTION-CRITICAL METADATA MISMATCH:\n' + '\n'.join(failures))
    critical = set(expected) | {'class_labels', 'n_features_in_'}
    differences = sorted(k for k in (set(embedded) | set(metadata)) - critical
                         if embedded.get(k, missing) != metadata.get(k, missing))
    if differences:
        logging.warning('Ignoring non-inference metadata differences: %s', ', '.join(differences))
    # Exercise the reused windowing and extraction, including output dtype/shape.
    probe = training.window_packets(np.ones((window_size, length), dtype=np.float32), window_size)
    features = inference_features(probe[0], variant)
    if features.shape != (len(names),) or features.dtype != np.float32 or not np.isfinite(features).all():
        raise ValueError('Live feature vector shape/dtype/finiteness does not match training')
    logging.info('Verified inference metadata, %d ordered features, %d-packet windows', len(names), window_size)
    return model, length, np.asarray([training.COORDINATES[c] for c in classes], dtype=float), classes, variant


def inference_features(window, variant):
    if variant == 'raw':
        return training.extract_features(window)
    if variant == 'relative_shape':
        from evaluate_normalization import variant_features
        return variant_features(window, 'relative_shape')
    raise ValueError(f'Unsupported preprocessing variant: {variant!r}')


class LineBuffer:
    """Retain timeout fragments; discard oversized lines THROUGH the next newline."""
    def __init__(self, limit=65536):
        self.limit = limit
        self.reset()

    def reset(self):
        self.pending = bytearray()
        self.discarding = False

    def feed(self, chunk):
        for part in chunk.splitlines(keepends=True):
            # splitlines also splits CR; only LF actually completes a CSI line.
            if not self.discarding:
                self.pending.extend(part)
                if len(self.pending) > self.limit:
                    self.pending.clear()
                    self.discarding = True
            if part.endswith(b'\n'):
                if not self.discarding:
                    yield self.pending.decode('utf-8', errors='replace')
                self.pending.clear()
                self.discarding = False


def udp_packet_to_csi_line(packet):
    """Present decoded UDP samples to the existing CSI_DATA parser."""
    values = ','.join(map(str, packet.iq))
    return f'CSI_DATA,{packet.sequence},udp,{packet.rssi},"[{values}]"\n'


class Display:
    def __init__(self, args):
        self.args = args
        self.history = deque(maxlen=args.history)
        self.plt = None
        if not args.no_gui:
            import matplotlib.pyplot as plt
            self.plt = plt
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set(xlim=(-3, 3), ylim=(0, 6), xlabel='X (ft)', ylabel='Y (ft)',
                        title='CSI localization — 6 ft × 6 ft')
            self.ax.set_aspect('equal')
            self.ax.set_xticks(range(-3, 4))
            self.ax.set_yticks(range(7))
            self.ax.grid(alpha=.3)
            for label, (x, y) in training.COORDINATES.items():
                self.ax.plot(x, y, '+', color='gray')
                self.ax.annotate(label, (x, y), xytext=(6, -14 if y == 6 else 6), textcoords='offset points')
            self.trail, = self.ax.plot([], [], '-', color='royalblue', alpha=.6)
            self.dot, = self.ax.plot([], [], 'o', markersize=12)
            self.status_text = self.fig.text(.02, .02, '', fontsize=10)
            self.fig.subplots_adjust(bottom=.23, top=.93)
            plt.show(block=False)

    def alive(self):
        return self.plt is None or self.plt.fignum_exists(self.fig.number)

    def pump(self):
        if self.plt and self.alive():
            self.fig.canvas.flush_events()

    def status(self, text):
        logging.warning(text)
        if self.plt and self.alive():
            self.status_text.set_text(text)
            self.status_text.set_color('crimson')
            self.dot.set_data([], [])
            self.fig.canvas.draw_idle()

    def reset(self):
        self.history.clear()
        if self.plt:
            self.trail.set_data([], [])
            self.dot.set_data([], [])

    def prediction(self, row):
        low = row['uncertain']
        message = (f"held class={row['predicted_position']} | top={row['top_position']} confidence={row['confidence']:.1%} "
                   f"raw=({row['raw_x_ft']:.2f}, {row['raw_y_ft']:.2f}) ft "
                   f"EMA=({row['smoothed_x_ft']:.2f}, {row['smoothed_y_ft']:.2f}) ft\n"
                   f"{row['packet_rate_hz']:.1f} packets/s | {row['update_rate_hz']:.2f} effective updates/s (stride {self.args.stride}) | "
                   f"rejected={row['rejected_packets']}" + (' | UNCERTAIN' if low else ''))
        if self.plt is None:
            print(row['timestamp_utc'], message.replace('\n', ' | '), flush=True)
            return
        self.history.append((row['smoothed_x_ft'], row['smoothed_y_ft']))
        xy = np.asarray(self.history)
        self.trail.set_data(xy[:, 0], xy[:, 1])
        self.dot.set_data([xy[-1, 0]], [xy[-1, 1]])
        self.dot.set_color('crimson' if low else 'forestgreen')
        self.status_text.set_text(message)
        self.status_text.set_color('crimson' if low else 'black')
        self.fig.canvas.draw_idle()

    def close(self):
        if self.plt:
            self.plt.close(self.fig)


class PacketDiagnostics:
    """Read-only cumulative counters; preprocessing failures count window attempts."""
    def __init__(self, enabled, transport='serial'):
        self.enabled = enabled
        self.counts = Counter({k: 0 for k in (
            'serial_lines', 'csi_data_lines', 'parsed_packets', 'corrupt_parse_rejects',
            'raw_length_rejects', 'preprocessing_rejects')})
        if transport == 'udp':
            self.counts.update({k: 0 for k in (
                'datagrams_received', 'malformed_datagrams', 'sequence_gaps',
                'duplicate_packets', 'out_of_order_packets')})
        self.latest_sequence = None
        self.recent_sequences = deque()
        self.recent_set = set()
        self.last_print = time.monotonic()

    def observe_udp_sequence(self, sequence):
        """Track arrivals with uint32 wraparound; counters never change packet order."""
        already_seen = sequence in self.recent_set
        if already_seen:
            self.counts['duplicate_packets'] += 1
        elif self.latest_sequence is not None:
            step = (sequence - self.latest_sequence) & 0xffffffff
            if step >= 0x80000000:
                self.counts['out_of_order_packets'] += 1
            elif step:
                self.counts['sequence_gaps'] += step - 1
                self.latest_sequence = sequence
        if self.latest_sequence is None:
            self.latest_sequence = sequence
        if not already_seen:
            if len(self.recent_sequences) == 4096:
                self.recent_set.remove(self.recent_sequences.popleft())
            self.recent_sequences.append(sequence)
            self.recent_set.add(sequence)

    def tick(self, buffer_length, now=None):
        now = time.monotonic() if now is None else now
        if self.enabled and now - self.last_print >= 1.0:
            print('PACKET DEBUG ' + ' '.join(f'{k}={v}' for k, v in self.counts.items())
                  + f' window_buffer_length={buffer_length}', flush=True)
            self.last_print = now


def run(args):
    model, length, coordinates, classes, variant = load_verified(Path(args.results).expanduser(), args.window_size, args.debug_metadata)
    logging.info('Live windows: %d packets, stride %d; threshold %.2f; effective rate measured over recent predictions',
                 args.window_size, args.stride, args.confidence_threshold)
    if args.transport == 'serial':
        import serial
    else:
        from firmware.csi_recv.tools.receive_udp import decode_packet
    if variant == 'relative_shape':
        import evaluate_normalization as packet_pipeline
    else:
        packet_pipeline = training
    display = Display(args)
    stream = log_file = None
    writer = None
    framer = LineBuffer()
    packets = []
    diagnostics = PacketDiagnostics(getattr(args, 'debug_packets', False), args.transport)
    ema = None
    accepted_class = None
    packet_times, update_times = deque(), deque()
    rejected = 0
    retry_at = 0.0
    last_packet = time.monotonic()
    stale = False
    fields = ['timestamp_utc', 'predicted_position', 'confidence', 'raw_x_ft', 'raw_y_ft',
              'smoothed_x_ft', 'smoothed_y_ft', 'packet_rate_hz', 'update_rate_hz',
              'rejected_packets', 'top_position', 'uncertain', 'stride_packets'] + [f'prob_{c}' for c in classes]
    try:
        if args.log is not None:
            path = Path(args.log or f"localization_{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%fZ}.csv").expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            log_file = path.open('x', newline='')
            writer = csv.DictWriter(log_file, fieldnames=fields)
            writer.writeheader()
            logging.info('Logging predictions to %s', path)
        while display.alive():
            display.pump()
            now = time.monotonic()
            diagnostics.tick(len(packets), now)
            if stream is None:
                if now < retry_at:
                    time.sleep(.02)
                    continue
                try:
                    if args.transport == 'serial':
                        stream = serial.Serial(args.port, args.baud, timeout=.05)
                        stream.reset_input_buffer()
                        connected = f'Connected {args.port} at {args.baud}'
                    else:
                        stream = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        stream.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
                        stream.bind((args.udp_bind, args.udp_port))
                        stream.settimeout(.05)
                        connected = f'Listening for CSI UDP on {args.udp_bind}:{args.udp_port}'
                    framer.reset()
                    packets.clear()
                    ema = None
                    accepted_class = None
                    display.reset()
                    packet_times.clear()
                    update_times.clear()
                    last_packet = now
                    stale = False
                    display.status(f'{connected}; collecting a full window…')
                except OSError as exc:
                    if stream is not None:
                        stream.close()
                    stream = None
                    label = 'Serial' if args.transport == 'serial' else 'UDP'
                    display.status(f'{label} unavailable: {exc}; retrying in {args.reconnect_delay:g}s')
                    retry_at = now + args.reconnect_delay
                    continue
            try:
                if args.transport == 'serial':
                    chunk = stream.read(min(max(stream.in_waiting, 1), 4096))
                    lines = framer.feed(chunk)
                else:
                    try:
                        datagram, _ = stream.recvfrom(65535)
                    except socket.timeout:
                        lines = ()
                    else:
                        diagnostics.counts['datagrams_received'] += 1
                        try:
                            packet = decode_packet(datagram)
                        except ValueError:
                            diagnostics.counts['malformed_datagrams'] += 1
                            rejected += 1
                            lines = ()
                        else:
                            diagnostics.observe_udp_sequence(packet.sequence)
                            # The existing parser and all downstream processing see
                            # the same CSI_DATA form as they do for serial input.
                            lines = (udp_packet_to_csi_line(packet),)
            except OSError as exc:
                label = 'Serial' if args.transport == 'serial' else 'UDP'
                display.status(f'{label} disconnected: {exc}; retrying…')
                stream.close()
                stream = None
                retry_at = time.monotonic() + args.reconnect_delay
                packets.clear()
                framer.reset()
                ema = None
                accepted_class = None
                continue
            for line in lines:
                if args.transport == 'serial':
                    diagnostics.counts['serial_lines'] += 1
                if 'CSI_DATA' in line:
                    diagnostics.counts['csi_data_lines'] += 1
                try:
                    iq = packet_pipeline.parse_csi_line(line)
                    if iq is None:
                        continue
                    diagnostics.counts['parsed_packets'] += 1
                    amplitude = packet_pipeline.iq_to_amplitude(iq)
                except (ValueError, OverflowError):
                    diagnostics.counts['corrupt_parse_rejects'] += 1
                    rejected += 1
                    continue
                if len(iq) != 2 * length:
                    diagnostics.counts['raw_length_rejects'] += 1
                    rejected += 1
                    continue
                now = time.monotonic()
                last_packet = now
                stale = False
                packet_times.append(now)
                packets.append(amplitude)
                if len(packets) < args.window_size:
                    continue
                window = packet_pipeline.window_packets(packets, args.window_size)[0]
                # Retain the overlap; only advance by accepted CSI packets.
                del packets[:args.stride]
                try:
                    features = inference_features(window, variant)
                except Exception:
                    if variant == 'relative_shape':
                        diagnostics.counts['preprocessing_rejects'] += 1
                    raise
                if features.shape != (model.n_features_in_,) or not np.isfinite(features).all():
                    if variant == 'relative_shape':
                        diagnostics.counts['preprocessing_rejects'] += 1
                    raise ValueError('LIVE FEATURE MISMATCH: wrong shape/nonfinite values')
                probabilities = np.asarray(model.predict_proba(features.reshape(1, -1))[0])
                if probabilities.shape != (9,) or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1):
                    raise ValueError('Invalid model class probabilities')
                raw = probabilities @ coordinates
                ema = raw.copy() if ema is None else args.alpha * raw + (1 - args.alpha) * ema
                update_times.append(now)
                for timestamps in (packet_times, update_times):
                    while timestamps and timestamps[0] < now - 5:
                        timestamps.popleft()
                def rate(timestamps):
                    return (len(timestamps) - 1) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 and timestamps[-1] > timestamps[0] else 0.0
                best = int(probabilities.argmax())
                uncertain = float(probabilities[best]) < args.confidence_threshold
                if not uncertain:
                    accepted_class = classes[best]
                row = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                           predicted_position=accepted_class or 'UNKNOWN', confidence=float(probabilities[best]),
                           top_position=classes[best], uncertain=uncertain, stride_packets=args.stride,
                           raw_x_ft=float(raw[0]), raw_y_ft=float(raw[1]),
                           smoothed_x_ft=float(ema[0]), smoothed_y_ft=float(ema[1]),
                           packet_rate_hz=rate(packet_times), update_rate_hz=rate(update_times), rejected_packets=rejected)
                row.update({f'prob_{c}': float(p) for c, p in zip(classes, probabilities)})
                display.prediction(row)
                if writer:
                    writer.writerow(row)
                    log_file.flush()
            if time.monotonic() - last_packet > args.stale_timeout and not stale:
                display.status('STALE: no valid CSI packets; clearing partial window and position history')
                packets.clear()
                framer.reset()
                ema = None
                accepted_class = None
                display.reset()
                packet_times.clear()
                update_times.clear()
                stale = True
    finally:
        if stream is not None:
            stream.close()
        if log_file:
            log_file.close()
        display.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', default='/home/komail-shah/csi_localization_results/')
    parser.add_argument('--transport', choices=('serial', 'udp'), default='serial',
                        help='CSI input transport (default: serial)')
    parser.add_argument('--port', default='/dev/ttyUSB0')
    parser.add_argument('--baud', type=int, default=921600)
    parser.add_argument('--udp-bind', default='0.0.0.0', help='Local IPv4 address for UDP listening')
    parser.add_argument('--udp-port', type=int, default=5005, help='RX destination UDP port (default: 5005)')
    parser.add_argument('--window-size', type=int, default=100, help='Must equal training metadata window size')
    parser.add_argument('--stride', type=int, default=25, help='Accepted packets between full-window predictions (default: 25)')
    parser.add_argument('--alpha', type=float, default=.3, help='EMA new-coordinate weight: (0, 1], 1 disables smoothing')
    parser.add_argument('--history', type=int, default=30)
    parser.add_argument('--confidence-threshold', '--low-confidence', dest='confidence_threshold', type=float, default=.45,
                        help='Below this probability hold the last confident class; XY still updates (default: 0.45)')
    parser.add_argument('--reconnect-delay', type=float, default=2)
    parser.add_argument('--stale-timeout', type=float, default=3)
    parser.add_argument('--log', nargs='?', const='', metavar='CSV', help='Log to CSV; omit filename for timestamped filename')
    parser.add_argument('--no-gui', action='store_true')
    parser.add_argument('--debug-metadata', action='store_true', help='Print artifact, JSON, and live values for every inference compatibility field')
    parser.add_argument('--debug-packets', action='store_true',
                        help='Print cumulative parse and transport counters plus buffer length once per second; preprocessing rejects count failed relative-shape window attempts')
    args = parser.parse_args()
    if not (args.baud > 0 and 1 <= args.udp_port <= 65535 and args.window_size >= 2 and 0 < args.alpha <= 1 and args.history > 0
            and 1 <= args.stride <= args.window_size and 0 <= args.confidence_threshold <= 1 and args.reconnect_delay > 0 and args.stale_timeout > 0):
        parser.error('Invalid baud/window/stride/alpha/history/confidence/timing setting')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        run(args)
    except KeyboardInterrupt:
        logging.info('Stopped by Ctrl+C')
    except Exception:
        logging.exception('Live localization failed')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
