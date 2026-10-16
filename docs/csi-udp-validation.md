# CSI UDP firmware validation — 2026-10-15

The updated `csi_recv` application was built with ESP-IDF 5.5.3 for ESP32-C5
and flashed to the confirmed receiver on `/dev/ttyUSB0`. The transmitter was
left running its existing firmware.

A 30-second serial observation measured **90.70 CSI packets/sec** from receiver
timestamps, with all packets queued. Before the laptop joined the receiver AP,
UDP send failures were counted while serial output continued normally.

The laptop then joined `CSI-RX` at `192.168.4.2` for a 60-second simultaneous
UDP/serial capture, using destination port 5005.

| Measurement | Result |
| --- | --- |
| Capture duration | 60.007 seconds |
| UDP packets received | 5,426 |
| UDP rate | 90.42 packets/sec |
| Serial packets during capture | 5,426 |
| UDP packets matching serial sequence, RSSI, and every CSI scalar | 5,426 |
| Mismatched packets / missing serial counterparts | 0 / 0 |
| Invalid UDP datagrams | 0 |
| Additional firmware drops during measurement | 0 |
| Queue drops | 0 |

The cumulative dropped counter stayed at 653 throughout the measured interval;
these were send failures before the laptop connected. The counters continued
printing every five seconds.

One serial record was interrupted by an ESP-IDF Wi-Fi association log before
the measurement interval. The existing multi-call `ets_printf` serial output
can interleave with driver logs. All serial records in the measured interval
were valid. The new statistics task is serialized against CSI printing.

The test script restored the original laptop Wi-Fi connection and removed its
temporary CSI network profile. The receiver remains flashed with the updated
firmware and advertises `CSI-RX`; ongoing UDP reception requires joining that
network again.

Additional checks passed: full ESP-IDF compile/link, three decoder unit tests,
a native C encoder/Python decoder comparison of all 2,048 supported scalar
values, and exact source comparisons of the original acquisition configuration
and serial formatting/conversion blocks.
