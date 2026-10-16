"""UDP transport checks that protect the serial-to-model contract."""
import struct
import unittest

import numpy as np

import evaluate_normalization
import realtime_localization
from firmware.csi_recv.tools.receive_udp import decode_packet


class RealtimeUdpTest(unittest.TestCase):
    def test_udp_uses_identical_relative_shape_parser_and_amplitude(self):
        samples = (-128, 127, -1, 0)
        header = struct.pack('!4sIbBBBHH', b'CSI1', 42, -63, 0, 11, 0, 4, 4)
        packet = decode_packet(header + struct.pack('!4h', *samples))
        line = realtime_localization.udp_packet_to_csi_line(packet)
        self.assertIn('CSI_DATA,42,udp,-63,', line)
        from_udp = evaluate_normalization.parse_csi_line(line)
        from_serial = evaluate_normalization.parse_csi_line(
            'CSI_DATA,42,1a:00:00:00:00:00,-63,11,0,0,0,0,11,0,0,0,4,0,"[-128,127,-1,0]"\n')
        np.testing.assert_array_equal(from_udp, from_serial)
        np.testing.assert_array_equal(
            evaluate_normalization.iq_to_amplitude(from_udp),
            evaluate_normalization.iq_to_amplitude(from_serial))

    def test_sequence_counters_include_wrap_and_classify_late_arrivals(self):
        diagnostics = realtime_localization.PacketDiagnostics(True, 'udp')
        for sequence in (0xfffffffe, 0, 0, 0xffffffff, 1, 0xffffffff):
            diagnostics.observe_udp_sequence(sequence)
        self.assertEqual(diagnostics.counts['sequence_gaps'], 1)
        self.assertEqual(diagnostics.counts['duplicate_packets'], 2)
        self.assertEqual(diagnostics.counts['out_of_order_packets'], 1)
        self.assertEqual(diagnostics.latest_sequence, 1)

    def test_serial_debug_counters_are_unchanged(self):
        diagnostics = realtime_localization.PacketDiagnostics(True)
        self.assertEqual(list(diagnostics.counts), [
            'serial_lines', 'csi_data_lines', 'parsed_packets',
            'corrupt_parse_rejects', 'raw_length_rejects', 'preprocessing_rejects'])


if __name__ == '__main__':
    unittest.main()
