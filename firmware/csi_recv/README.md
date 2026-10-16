# ESP32-C5 CSI receiver with UDP forwarding

Based on the installed Espressif `esp-csi/examples/get-started/csi_recv`.
The acquisition configuration, source MAC filter, gain compensation, ESP-NOW
settings, channel 11, HT40 bandwidth, and serial `CSI_DATA` fields and values
are preserved. The transmitter firmware and its packet interval are unchanged.

The RX creates a WPA2 SoftAP on the existing channel. The laptop joins this AP;
no router or upstream Internet connection is needed. This avoids station scans
and router association changing the CSI channel. AP beacons and UDP traffic
necessarily add radio airtime. See Espressif's [Wi-Fi driver documentation](https://docs.espressif.com/projects/esp-idf/en/release-v5.5/esp32c5/api-guides/wifi.html)
for AP/STA coexistence and channel behavior.

## Configure and build

Use ESP-IDF **5.5.3** and target **esp32c5**. From this directory:

```bash
source /home/komail-shah/.espressif/tools/activate_idf_v5.5.3.sh
idf.py set-target esp32c5  # only for a fresh checkout; resets existing sdkconfig
idf.py menuconfig
idf.py build
idf.py -p /dev/ttyACM0 flash monitor
```

Under **CSI UDP forwarding**, set:

- `CSI_UDP_DEST_IP`: laptop IPv4 on the RX AP, default `192.168.4.2`.
- `CSI_UDP_DEST_PORT`: laptop UDP port, default `5005`.
- `CSI_UDP_AP_SSID`: default `CSI-RX`.
- `CSI_UDP_AP_PASSWORD`: default `csi-receiver`; use your own 8–63 byte password.

Connect the laptop to `CSI-RX`. The RX DHCP server uses the default
`192.168.4.1/24` network. Check the laptop's actual address with `ip -4 addr`;
set the destination to that address, or configure the laptop statically as
`192.168.4.2/24`. Permit inbound UDP on the chosen port in the laptop firewall.
Keep the existing serial monitor baud rate of **921600**.

Start the laptop receiver (Python 3.10+, standard library only):

```bash
python3 tools/receive_udp.py --port 5005
# Save complete arrays; rate reports go to stderr:
python3 tools/receive_udp.py --port 5005 --json > csi.jsonl
```

## Callback, serial output, and counters

The Wi-Fi callback retains the original gain tracking at reception time,
snapshots metadata and the entire driver-owned CSI buffer, and calls
`xQueueSend(..., 0)`. It never allocates, prints, waits for queue space, or sends
network data. The queue holds 32 packets, each with space for 2048 raw CSI bytes
(about 70 KiB total). Buffers above that bound are dropped and counted, never
truncated. The current C5 HT20/HT40 acquisition fits within this bound.

A priority-4 worker prints the original gain log and complete original
`CSI_DATA` record, then sends one nonblocking UDP datagram. Socket setup failure
is retried; each failed datagram is counted and the stream continues. There
are no UDP retries, pacing delays, or changes to the transmitter's sampling
rate. Serial output continues when the laptop is absent or UDP sends fail.
A full queue drops the incoming packet from both outputs; it does not block
the Wi-Fi task. Counters expose this overload condition.

A separate task reports cumulative `CSI_STATS` every five seconds:

- `received`: callbacks matching the existing source MAC with a non-null CSI buffer.
- `queued`: packets successfully copied into the worker queue.
- `udp_sent`: complete datagrams accepted by the local UDP socket.
- `dropped`: invalid/oversize input, full queue, or unsuccessful UDP send; one
  increment per packet that cannot be forwarded.

Counters are protected by short critical sections and wrap at 2^32. While
packets are pending, `received - udp_sent - dropped` includes in-flight packets.
A serial mutex prevents the statistics task from inserting its report into the
middle of this firmware's multi-call `CSI_DATA` line. UDP socket success does
not acknowledge laptop delivery: UDP loss/reordering is detected on the laptop
using transmitter sequence numbers. The sequence is the same payload ID already
printed by serial, not a new counter that would hide upstream losses.

## UDP wire format (CSI1)

Each datagram contains a 16-byte header followed by the complete compensated
CSI array. Multibyte integers use **network (big-endian) byte order**. No C
struct padding is transmitted.

| Offset | Bytes | Meaning |
| --- | --- | --- |
| 0 | 4 | ASCII `CSI1` (protocol version 1) |
| 4 | 4 | Unsigned transmitter sequence number, same as serial `seq` |
| 8 | 1 | Signed RSSI in dBm |
| 9 | 1 | `first_word_invalid` flag; invalid entries are retained |
| 10 | 1 | Received channel |
| 11 | 1 | Reserved, zero |
| 12 | 2 | Original raw CSI byte length |
| 14 | 2 | Number of signed 16-bit CSI scalar values following |
| 16 | 2 × count | Signed 16-bit values in precisely the serial array order |

Values use the exact existing `(int16_t)(compensate_gain * info->buf[i])`
conversion. No I/Q reordering, normalization, filtering, or truncation is added.
Signed 16-bit transport preserves compensated values that may exceed int8.
The existing `CSI_FORCE_LLTF=0` is unchanged: count equals raw length. The
firmware retains the original compile-time LLTF branch; the supplied Python
receiver intentionally validates the current default format.

Typical HT40 arrays fit one Ethernet/Wi-Fi IP MTU. Datagrams over the link MTU
require IP fragmentation, so larger future CSI configurations need appropriate
laptop receive/reassembly support. Maximum supported datagram is 4112 bytes.

## Validation and throughput

Run `python3 -m unittest discover -s tools -p 'test_*.py'` for decoder checks.
Build with the installed ESP-IDF 5.5.3 toolchain for a compile/link check.

For hardware acceptance, run the unchanged TX and both serial/UDP capture for
at least 60 seconds. Expect approximately 450 received/queued/sent packets per
five-second interval (about 90/s), no growing drops, and the same laptop rate.
Match serial and UDP packets by sequence and compare RSSI and every CSI scalar.
Disconnect/reconnect the laptop and verify serial continues, statistics keep
printing, and UDP resumes. A successful build alone does not establish the
hardware packet rate or radio coexistence behavior.
