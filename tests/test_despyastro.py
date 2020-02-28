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
import numpy as np

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

    def test_eq2gal(self):
        long = 25.0
        lat = 15.0
        new_long, new_lat = cds.eq2gal(long, lat)
        self.assertNotAlmostEqual(long, new_long[0], 5)
        self.assertNotAlmostEqual(lat, new_lat[0], 5)
        back_long, back_lat = cds.gal2eq(new_long, new_lat)
        self.assertAlmostEqual(long, back_long[0], 5)
        self.assertAlmostEqual(lat, back_lat[0], 5)

    def test_eq2gal_1950(self):
        long = 25.0
        lat = 15.0
        new_long, new_lat = cds.eq2gal(long, lat, True)
        self.assertNotAlmostEqual(long, new_long[0], 5)
        self.assertNotAlmostEqual(lat, new_lat[0], 5)
        back_long, back_lat = cds.gal2eq(new_long, new_lat, True)
        self.assertAlmostEqual(long, back_long[0], 5)
        self.assertAlmostEqual(lat, back_lat[0], 5)

    def test_eq2ec(self):
        long = 25.0
        lat = 15.0
        new_long, new_lat = cds.eq2ec(long, lat)
        self.assertNotAlmostEqual(long, new_long[0], 5)
        self.assertNotAlmostEqual(lat, new_lat[0], 5)
        back_long, back_lat = cds.ec2eq(new_long, new_lat)
        self.assertAlmostEqual(long, back_long[0], 5)
        self.assertAlmostEqual(lat, back_lat[0], 5)

    def test_eq2ec_1950(self):
        long = 25.0
        lat = 15.0
        new_long, new_lat = cds.eq2ec(long, lat, True)
        self.assertNotAlmostEqual(long, new_long[0], 5)
        self.assertNotAlmostEqual(lat, new_lat[0], 5)
        back_long, back_lat = cds.ec2eq(new_long, new_lat, True)
        self.assertAlmostEqual(long, back_long[0], 5)
        self.assertAlmostEqual(lat, back_lat[0], 5)

    def test_ec2gal(self):
        long = 25.0
        lat = 15.0
        new_long, new_lat = cds.ec2gal(long, lat)
        self.assertNotAlmostEqual(long, new_long[0], 5)
        self.assertNotAlmostEqual(lat, new_lat[0], 5)
        back_long, back_lat = cds.gal2ec(new_long, new_lat)
        self.assertAlmostEqual(long, back_long[0], 5)
        self.assertAlmostEqual(lat, back_lat[0], 5)

    def test_ec2gal_1950(self):
        long = 25.0
        lat = 15.0
        new_long, new_lat = cds.ec2gal(long, lat, True)
        self.assertNotAlmostEqual(long, new_long[0], 5)
        self.assertNotAlmostEqual(lat, new_lat[0], 5)
        back_long, back_lat = cds.gal2ec(new_long, new_lat, True)
        self.assertAlmostEqual(long, back_long[0], 5)
        self.assertAlmostEqual(lat, back_lat[0], 5)

    def test_thetaphi2xyz(self):
        theta = 1.225
        phi = -.25
        x, y, z = cds._thetaphi2xyz(theta, phi)

        new_theta, new_phi = cds._xyz2thetaphi(x, y, z)

        self.assertAlmostEqual(theta, new_theta, 5)
        self.assertAlmostEqual(phi, new_phi, 5)

    def test_eq2xyz(self):
        long = 25.0
        lat = 15.0
        x, y, z = cds.eq2xyz(long, lat)

        back_long, back_lat = cds.xyz2eq(x, y, z)
        self.assertAlmostEqual(long, back_long[0], 5)
        self.assertAlmostEqual(lat, back_lat[0], 5)

    def test_sphdist(self):
        dist = cds.sphdist(15., -15., 25., 30.)
        self.assertAlmostEqual(46.020717, dist[0], 5)

    def test_gcirc(self):
        dist = cds.gcirc(15.,-15.,25.,30.)
        self.assertAlmostEqual(dist[0], 0.803213, 5)
        dist = cds.gcirc(15.,-15.,25.,30., True)
        self.assertAlmostEqual(dist[0][0], 0.803213, 5)
        self.assertAlmostEqual(dist[1][0], 1.360259, 5)

    def test_atbound(self):
        low = 0.
        hi = 360.
        d = np.array(range(-100,400, 10), dtype='f8')
        #print(d)
        self.assertTrue(len(np.where(d < low)[0]) > 0)
        self.assertTrue(len(np.where(d > hi)[0]) > 0)
        cds.atbound(d, low, hi)
        #print(d)
        self.assertTrue(len(np.where(d < low)[0]) == 0)
        self.assertTrue(len(np.where(d > hi)[0]) == 0)

        new_d = copy.deepcopy(d)
        cds.atbound(d, low, hi)
        self.assertTrue(d.all() == new_d.all())

    def test_atbound2(self):
        theta = np.array(range(-300,400, 10), dtype='f8')
        phi = np.array(range(-300,400, 10), dtype='f8')

        self.assertTrue(len(np.where(theta < -180.)[0]) > 0)
        self.assertTrue(len(np.where(theta > 180.)[0]) > 0)
        self.assertTrue(len(np.where(phi < 0.)[0]) > 0)
        self.assertTrue(len(np.where(phi > 360.)[0]) > 0)

        cds.atbound2(theta, phi)

        self.assertTrue(len(np.where(theta < -180.)[0]) == 0)
        self.assertTrue(len(np.where(theta > 180.)[0]) == 0)
        self.assertTrue(len(np.where(phi < 0.)[0]) == 0)
        self.assertTrue(len(np.where(phi > 360.)[0]) == 0)

        theta = np.array(range(-30,40, 10), dtype='f8')
        phi = np.array(range(0,40, 5), dtype='f8')

        t_theta = copy.deepcopy(theta)
        t_phi = copy.deepcopy(phi)

        cds.atbound2(theta, phi)
        self.assertTrue(np.array_equal(theta, t_theta))
        self.assertTrue(np.array_equal(phi, t_phi))

    def test_eq2sdss_errors(self):
        ra = [0, 1]
        dec = [5, 10, 15]
        self.assertRaises(ValueError, cds.eq2sdss, ra, dec)

        ra = [-5, 10]
        dec = [-5, 10]
        self.assertRaises(ValueError, cds.eq2sdss, ra, dec)

        ra = [375, 10]
        self.assertRaises(ValueError, cds.eq2sdss, ra, dec)

        ra = [10, 50]
        dec = [-95,82]
        self.assertRaises(ValueError, cds.eq2sdss, ra, dec)

        dec = [-82, 99]
        self.assertRaises(ValueError, cds.eq2sdss, ra, dec)

    def test_eq2sdss(self):
        ra = [10, 20]
        dec = [-30, 50]
        clambda, ceta = cds.eq2sdss(ra, dec)
        self.assertAlmostEqual(clambda[0], -4.32875, 5)
        self.assertAlmostEqual(ceta[1], 96.52504, 5)

    def test_sdss2eq_errors(self):
        clambda = [-99, 85]
        ceta = [10, 20]
        self.assertRaises(ValueError, cds.sdss2eq, clambda, ceta)

        clambda = [-85, 99]
        self.assertRaises(ValueError, cds.sdss2eq, clambda, ceta)

        clambda = [-85, 62]
        ceta = [-190, 125]
        self.assertRaises(ValueError, cds.sdss2eq, clambda, ceta)

        ceta = [-125, 190]
        self.assertRaises(ValueError, cds.sdss2eq, clambda, ceta)

    def test_sdss2eq(self):
        clambda = [-4.32875, -9.57658]
        ceta = [177.59467, 96.52504]

        ra, dec = cds.sdss2eq(clambda, ceta)
        self.assertAlmostEqual(ra[0], 10.0, 5)
        self.assertAlmostEqual(dec[1], 50.0, 5)

    def test_eq2survey_errors(self):
        ra = [0, 1]
        dec = [5, 10, 15]
        self.assertRaises(ValueError, cds._eq2survey, ra, dec)

        ra = [-5, 10]
        dec = [-5, 10]
        self.assertRaises(ValueError, cds._eq2survey, ra, dec)

        ra = [375, 10]
        self.assertRaises(ValueError, cds._eq2survey, ra, dec)

        ra = [10, 50]
        dec = [-95,82]
        self.assertRaises(ValueError, cds._eq2survey, ra, dec)

        dec = [-82, 99]
        self.assertRaises(ValueError, cds._eq2survey, ra, dec)

    def test_eq2survey(self):
        ra = [10, 20]
        dec = [-30, 50]
        lambda_, eta = cds._eq2survey(ra, dec)
        self.assertAlmostEqual(lambda_[0], -175.67125, 5)
        self.assertAlmostEqual(eta[1], -83.47496, 5)

    def test_dec_parse(self):
        dec_str = '5:25:30.8'
        dec = cds.dec_parse(dec_str)
        self.assertAlmostEqual(dec, 5.425222, 5)

        #dec_str = '-5:25:30.8'
        #ndec = cds.dec_parse(dec_str)
        #self.assertAlmostEqual(-1. * dec, ndec, 9)

        dec_str = '5'
        self.assertEqual(5., cds.dec_parse(dec_str))

        dec_str = '5:30'
        self.assertEqual(5.5, cds.dec_parse(dec_str))

    def test_ra_parse(self):
        ra_str = '05:25:22'
        ra = cds.ra_parse(ra_str)
        self.assertAlmostEqual(ra, 75.422778, 5)

        ra = cds.ra_parse(ra_str, False)
        self.assertAlmostEqual(ra, 5.422778, 5)

        ra_str = '5'
        self.assertEqual(75., cds.ra_parse(ra_str))

        ra_str = '5:30'
        self.assertEqual(75.5, cds.ra_parse(ra_str))
if __name__ == '__main__':
    unittest.main()
