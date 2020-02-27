#!/usr/bin/env python3

import unittest
import os
import shutil
import stat
import sys
import copy
import time
import errno
from contextlib import contextmanager
from collections import OrderedDict
from io import StringIO
from mock import patch

from despyastro import coords as cds

class TestCoords(unittest.TestCase):
    def test_euler_J2000_galactic(self):
        long = 25.0
        lat = 15.0
        new_long, new_lat = cds.euler(long, lat, 1)
        self.assertNotAlmostEqual(long, new_long[0], 5)
        self.assertNotAlmostEqual(lat, new_lat[0], 5)
        back_long, back_lat = cds.euler(new_long, new_lat, 2)
        self.assertAlmostEqual(long, back_long[0], 5)
        self.assertAlmostEqual(lat, back_lat[0], 5)

    def test_euler_B1950_galactic(self):
        long = 25.0
        lat = 15.0
        new_long, new_lat = cds.euler(long, lat, 1, True)
        self.assertNotAlmostEqual(long, new_long[0], 5)
        self.assertNotAlmostEqual(lat, new_lat[0], 5)
        back_long, back_lat = cds.euler(new_long, new_lat, 2, True)
        self.assertAlmostEqual(long, back_long[0], 5)
        self.assertAlmostEqual(lat, back_lat[0], 5)

    def test_euler_J2000_ecliptic(self):
        long = 25.0
        lat = 15.0
        new_long, new_lat = cds.euler(long, lat, 3)
        self.assertNotAlmostEqual(long, new_long[0], 5)
        self.assertNotAlmostEqual(lat, new_lat[0], 5)
        back_long, back_lat = cds.euler(new_long, new_lat, 4)
        self.assertAlmostEqual(long, back_long[0], 5)
        self.assertAlmostEqual(lat, back_lat[0], 5)

    def test_euler_B1950_ecliptic(self):
        long = 25.0
        lat = 15.0
        new_long, new_lat = cds.euler(long, lat, 3, True)
        self.assertNotAlmostEqual(long, new_long[0], 5)
        self.assertNotAlmostEqual(lat, new_lat[0], 5)
        back_long, back_lat = cds.euler(new_long, new_lat, 4, True)
        self.assertAlmostEqual(long, back_long[0], 5)
        self.assertAlmostEqual(lat, back_lat[0], 5)

    def test_euler_ecliptic_galactic_J2000(self):
        long = 25.0
        lat = 15.0
        new_long, new_lat = cds.euler(long, lat, 5)
        self.assertNotAlmostEqual(long, new_long[0], 5)
        self.assertNotAlmostEqual(lat, new_lat[0], 5)
        back_long, back_lat = cds.euler(new_long, new_lat, 6)
        self.assertAlmostEqual(long, back_long[0], 5)
        self.assertAlmostEqual(lat, back_lat[0], 5)

    def test_euler_ecliptic_galactic_B1950(self):
        long = 25.0
        lat = 15.0
        new_long, new_lat = cds.euler(long, lat, 5, True)
        self.assertNotAlmostEqual(long, new_long[0], 5)
        self.assertNotAlmostEqual(lat, new_lat[0], 5)
        back_long, back_lat = cds.euler(new_long, new_lat, 6, True)
        self.assertAlmostEqual(long, back_long[0], 5)
        self.assertAlmostEqual(lat, back_lat[0], 5)


if __name__ == '__main__':
    unittest.main()
