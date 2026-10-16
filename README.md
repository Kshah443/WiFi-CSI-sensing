# Wi-Fi CSI Activity Sensing

A Wi-Fi Channel State Information (CSI) sensing project using ESP32-C5 hardware
and machine learning for device-free human activity classification.

## Current Capabilities

The current system classifies three conditions:

- Empty room
- Person standing still
- Person walking

## Baseline Results

- 29,996 valid CSI packets
- 15 total recorded trials
- 294 usable windows
- 12 training trials
- 3 held-out test trials
- 238 training windows
- 56 test windows
- Random Forest test accuracy: **96.43%**

The train/test split is performed by entire trial rather than by individual
windows, helping reduce leakage between training and test data.

## Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

## Hardware

- 2× ESP32-C5 development boards
- External dual-band Wi-Fi antennas
- Old laptop running Ubuntu 26.04.1 LTS

## Pipeline

ESP32-C5 TX
→ Wi-Fi channel
→ ESP32-C5 RX
→ CSI capture
→ preprocessing
→ feature extraction
→ Random Forest classifier

## Receiver firmware

The ESP32-C5 receiver now supports queued UDP forwarding alongside the existing
serial `CSI_DATA` output. See [firmware setup and UDP format](firmware/csi_recv/README.md)
for ESP-IDF 5.5.3 build instructions, laptop networking, and packet-rate checks.

For live localization over that link, join the laptop to the receiver's `CSI-RX`
network and run:

```bash
.venv/bin/python realtime_localization.py --transport udp --udp-port 5005 \
  --results /home/komail-shah/csi_localization_relative_results \
  --no-gui --debug-packets --log
```

The UDP port must match the receiver's configured destination port. UDP defaults
to listening on all IPv4 interfaces; use `--udp-bind` to restrict it. With no
`--transport` option, live localization still reads serial at `/dev/ttyUSB0`.
`--debug-packets` shows UDP datagrams, malformed datagrams, sequence gaps,
duplicates, and out-of-order packets alongside the existing parsing counters.
The received 16-bit CSI values go through the existing serial parser, including
its signed-int8 range check, so model preprocessing and CSV columns stay the same.

## Current Status

The first working prototype successfully captures CSI in real time and
distinguishes empty-room, stationary-person, and walking conditions.

## Limitations

The current baseline was trained and evaluated on a small dataset collected
in a single environment with a fixed transmitter/receiver arrangement.

Although testing was performed using entirely held-out trial files rather than
randomly splitting windows, the reported 96.43% accuracy should be treated as
an initial baseline rather than evidence of cross-room or cross-user
generalization.

Future testing will evaluate performance across different sensor positions,
room layouts, subjects, walking directions, and environmental conditions.

Next steps:

- Testing generalization across different TX/RX positions
- Testing different walking directions and speeds
- expand to multi-node sensing(three or four boards) if additional spatial coverage is needed
- Real-time inference
- Comparing Random Forest with neural network models

## Third-Party Software

This project uses Espressif ESP-CSI and ESP-IDF.

ESP-CSI:
https://github.com/espressif/esp-csi

ESP-IDF:
https://github.com/espressif/esp-idf

Third-party code remains subject to its original license.
