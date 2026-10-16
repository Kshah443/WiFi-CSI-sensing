import struct
import unittest

from receive_udp import decode_packet


class DecodePacketTest(unittest.TestCase):
    def test_known_packet_signed_values_and_sequence(self):
        # A wire fixture, including compensated values outside the int8 range.
        data = bytes.fromhex("43534931 fffffffe c4 01 0b 00 0006 0006 "
                             "8000 ff00 ffff 0000 0100 7fff")
        packet = decode_packet(data)
        self.assertEqual(packet.sequence, 0xFFFFFFFE)
        self.assertEqual(packet.rssi, -60)
        self.assertTrue(packet.first_word_invalid)
        self.assertEqual(packet.channel, 11)
        self.assertEqual(packet.raw_length, 6)
        self.assertEqual(packet.iq, (-32768, -256, -1, 0, 256, 32767))

    def test_maximum_array_is_preserved(self):
        values = tuple(range(-1024, 1024))
        data = struct.pack("!4sIbBBBHH", b"CSI1", 1, -99, 0, 11, 0, 2048, 2048)
        data += struct.pack("!2048h", *values)
        self.assertEqual(decode_packet(data).iq, values)

    def test_truncated_extra_and_invalid_datagrams_rejected(self):
        valid = bytes.fromhex("43534931 00000001 c4 00 0b 00 0002 0002 0001 ffff")
        variants = [valid[:8], valid[:-1], valid + b"\x00"]
        for offset, value in [(0, 0), (9, 2), (11, 1), (13, 0), (15, 1)]:
            invalid = bytearray(valid)
            invalid[offset] = value
            variants.append(bytes(invalid))
        for data in variants:
            with self.subTest(data=data.hex()), self.assertRaises(ValueError):
                decode_packet(data)


if __name__ == "__main__":
    unittest.main()
