#!/usr/bin/env python3

import unittest
import os
import stat
import math
import stat
import sys
import copy
import time
import logging
import errno
from contextlib import contextmanager
from collections import OrderedDict
from io import StringIO
from mock import patch
import mock
import numpy as np
from astropy.io import fits
import fitsio

from despyastro import coords as cds
from despyastro import genutil as gu
from despyastro import tableio as tio
from despyastro import wcsutil as wcs
from despyastro import astrometry as ast
from despyastro import CCD_corners as ccd
from despyastro import zipper_interp as zpi

import despydmdb.desdmdbi as dmdbi
from MockDBI import MockConnection

ROOT = '/var/lib/jenkins/test_data/'

@contextmanager
def capture_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


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

        long = 0.15
        lat = 1.081
        x, y, z = cds.eq2xyz(long, lat, units='rad')

        back_long, back_lat = cds.xyz2eq(x, y, z, units='rad')
        self.assertAlmostEqual(long, back_long[0], 5)
        self.assertAlmostEqual(lat, back_lat[0], 5)


    def test_sphdist(self):
        dist = cds.sphdist(15., -15., 25., 30.)
        self.assertAlmostEqual(46.020717, dist[0], 5)

        dist = cds.sphdist(0.25, -0.8, 1.2, 0.1, units=['rad','rad'])
        self.assertAlmostEqual(1.232774, dist[0], 5)

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

    def test_fitsheader2dict(self):
        header = fits.Header()
        header.append(('NAME', 'hello'))
        header.append(('ra', 3.65))

        hdr = cds.fitsheader2dict(header)
        self.assertEqual(3.65, hdr['ra'])
        self.assertEqual('hello', hdr['name'])

    def test_shiftlon(self):
        long = [10., 50., 385]
        shifted = cds.shiftlon(long)

        self.assertEqual(shifted[0], 10.)
        self.assertEqual(shifted[2], 25.)

        shifted = cds.shiftlon(long, wrap=False)
        self.assertEqual(shifted[0], 10.)
        self.assertEqual(shifted[2], 385.)

        shifted = cds.shiftlon(long, 20.)
        self.assertEqual(shifted[0], 350.)
        self.assertEqual(shifted[2], 365.)

        shifted = cds.shiftlon(long, -20.)
        self.assertEqual(shifted[0], 30.)
        self.assertEqual(shifted[2], 45.)

    def test_shiftra(self):
        long = [10., 50., 385]
        shifted = cds.shiftra(long)

        self.assertEqual(shifted[0], 10.)
        self.assertEqual(shifted[2], 25.)

        shifted = cds.shiftra(long, wrap=False)
        self.assertEqual(shifted[0], 10.)
        self.assertEqual(shifted[2], 385.)

        shifted = cds.shiftra(long, 20.)
        self.assertEqual(shifted[0], 350.)
        self.assertEqual(shifted[2], 365.)

        shifted = cds.shiftra(long, -20.)
        self.assertEqual(shifted[0], 30.)
        self.assertEqual(shifted[2], 45.)

    def test_radec2aitoff(self):
        r, d = cds.radec2aitoff(5.743, -12.15)
        self.assertAlmostEqual(r[0], 6.27052, 5)
        self.assertAlmostEqual(d[0], -13.474167, 5)

    def test_check_range(self):
        allowed = [0., 180.]
        rng = cds._check_range(None, allowed)
        self.assertEqual(rng, allowed)

        rng = cds._check_range([10., 20.], allowed)
        self.assertEqual(rng, [10., 20.])

    def test_check_range_errors(self):
        self.assertRaises(ValueError, cds._check_range, 5, [10, 20])

        self.assertRaises(ValueError, cds._check_range, [0, 15], [10, 20])

        self.assertRaises(ValueError, cds._check_range, [15, 25], [10, 20])

    def test_randsphere(self):
        r, d = cds.randsphere(155, [0, 15], [-5, 5])
        self.assertEqual(0, len(np.where(r < 0)[0]))
        self.assertEqual(len(r), len(d))
        self.assertEqual(len(r), 155)
        self.assertEqual(155, len(np.where(r <= 15)[0]))

        self.assertEqual(0, len(np.where(d < -5)[0]))
        self.assertEqual(155, len(np.where(d <= 5)[0]))

        x, y, z = cds.randsphere(155, [0, 15], [-5, 5], system='xyz')
        self.assertEqual(len(x), len(z))
        self.assertEqual(len(x), len(y))
        self.assertEqual(len(x), 155)

        temp = np.ones(155)
        w = x*x + y*y + z*z
        self.assertTrue(np.allclose((x*x + y*y + z*z), temp, atol=0.00001))

    def test_randcap(self):
        r, d = cds.randcap(155, 22.5, -10, 10)
        r = r - 22.5
        d = d + 10
        temp = np.sqrt(r*r + d*d)
        self.assertTrue(max(temp) < 10.4) # to accound for testing in xyz rather than sperical

        r, d, rad = cds.randcap(155, 22.5, -10, 10, True)
        self.assertEqual(len(rad), 155)
        rad = np.rad2deg(rad)
        self.assertTrue(max(rad) <= 10.0)

    def test_rect_area(self):
        sky = 4. * 180. * 180. / math.pi

        area = cds.rect_area(0., 5., 0., 5.)
        self.assertTrue(area < 25.)
        self.assertTrue(area > (0.99 * 25.))

        self.assertAlmostEqual(sky / 8., cds.rect_area(0., 90., 0., 90.), 5)



class TestGenUtil(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sfile = 'services.ini'
        open(cls.sfile, 'w').write("""

[db-maximal]
PASSWD  =   maximal_passwd
name    =   maximal_name_1    ; if repeated last name wins
user    =   maximal_name      ; if repeated key, last one wins
Sid     =   maximal_sid       ;comment glued onto value not allowed
type    =   POSTgres
server  =   maximal_server

[db-minimal]
USER    =   Minimal_user
PASSWD  =   Minimal_passwd
name    =   Minimal_name
sid     =   Minimal_sid
server  =   Minimal_server
type    =   oracle

[db-test]
USER    =   Minimal_user
PASSWD  =   Minimal_passwd
name    =   Minimal_name
sid     =   Minimal_sid
server  =   Minimal_server
type    =   test
port    =   0
""")
        os.chmod(cls.sfile, (0xffff & ~(stat.S_IROTH | stat.S_IWOTH | stat.S_IRGRP | stat.S_IWGRP)))
        cls.dbh = dmdbi.DesDmDbi(cls.sfile, 'db-test')

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.sfile)
        MockConnection.destroy()

    def test_query2dict_of_columns(self):
        query = "select * from image"
        res = gu.query2dict_of_columns(query, self.dbh)
        self.assertTrue('FILENAME' in res.keys())
        self.assertTrue('PV1_9' in res.keys())
        self.assertEqual(7257, len(res['FILENAME']))
        self.assertTrue(isinstance(res['FILENAME'], tuple))

        res = gu.query2dict_of_columns(query, self.dbh, True)
        self.assertTrue('FILENAME' in res.keys())
        self.assertTrue('PV1_9' in res.keys())
        self.assertEqual(7257, len(res['FILENAME']))
        self.assertFalse(isinstance(res['FILENAME'], tuple))
        self.assertTrue(res['FILENAME'].dtype == np.object)

    def test_query2rec(self):
        query = "select * from image"
        res = gu.query2rec(query, self.dbh)
        self.assertTrue(isinstance(res, np.recarray))
        self.assertEqual(7257, len(res['FILENAME']))

        query = "select * from exposure"
        res = gu.query2rec(query, self.dbh)
        self.assertFalse(res)

        with capture_output() as (out, _):
            res = gu.query2rec(query, self.dbh, True)
            self.assertFalse(res)
            output = out.getvalue().strip()
            self.assertTrue("returned no results" in output)

class TestTableIO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dfile = 'test.dat'
        cls.data = (('#header', 'name1', 'name2', '', 'name3', 'name4'),
                    ('item', '1', '2', '', '4', '9'),
                    ('', 'abc', 'ggh', '', 'hhy', 'hhfb'),
                    ('',' def|jkl', 'qwe|lkj', '', 'kpi|mnb', 'kel|bnm'))

    @classmethod
    def setUp(cls):
        open(cls.dfile, 'w').write('')

    @classmethod
    def tearDownClass(cls):
        #pass
        os.unlink(cls.dfile)

    def test_header(self):
        b = tio.get_header(self.dfile)
        self.assertFalse(b)

        tio.put_header(self.dfile, "this is a header\n")
        tio.put_header(self.dfile, "# this is also a header")
        tio.put_header(self.dfile, '')
        open(self.dfile, 'a').write("not a header\n# also not a header\n")
        b = tio.get_header(self.dfile)

        self.assertTrue('this is a header' in b)
        self.assertTrue('this is also a header' in b)
        self.assertFalse('not a header' in b)
        #os.unlink('test.dat')

    def test_str(self):
        bfile = 'bad.dat'

        bad_data = (('a','b','c','d'),
                    ('1', '2'))

        self.assertRaises(Exception, tio.put_str, bfile, bad_data)

        self.assertRaises(Exception, tio.put_str, bfile, [])
        try:
            os.unlink(bfile)
        except:
            pass
        tio.put_str(self.dfile, self.data)

        res = tio.get_str(self.dfile, cols=[1,2,3])

        self.assertEqual(4, len(res[0]))
        self.assertEqual(res[1][0], 'abc')
        self.assertEqual(res[0][1], '2')

        res = tio.get_str(self.dfile, cols=[0,1], nrows=2, sep='|')
        self.assertEqual(2, len(res[0]))
        self.assertEqual(res[1][0], 'jkl')
        self.assertEqual(res[0][1], 'name2 2 ggh qwe')

        res = tio.get_str(self.dfile, cols=1)

        self.assertEqual(4, len(res))
        self.assertEqual(res[0][0], '1')
        #os.unlink(dfile)

    def test_str_as_list(self):
        #dfile = 'test.dat'
        tio.put_str(self.dfile, self.data)

        lines = open(self.dfile, 'r').readlines()

        res = tio.get_str(lines, cols=1)
        self.assertEqual(4, len(res))
        self.assertEqual(res[0][0], '1')


    def test_data(self):
        self.assertRaises(Exception, tio.put_data, self.dfile, [])
        data = ((18.156, 22.556, 29.4, 186.25, -0.0000003365),
                (1, 2, 0, 4, 9),
                (3.556, 4.98776, 6.23765, 2.15, 16.7))
        tio.put_data(self.dfile, data, "Here is a header")
        res = tio.get_data(self.dfile)

        self.assertAlmostEqual(data[0][0], res[0], 3)
        self.assertAlmostEqual(data[0][4], res[4], 3)

        res = tio.get_data(self.dfile, cols=[1,2])
        self.assertEqual(data[1][0], res[0][0])
        self.assertAlmostEqual(data[2][2], res[1][2], 5)

        self.assertRaises(IndexError, tio.get_data, self.dfile, cols=[2,3,4])

        tio.put_data(self.dfile, data)

        new_data = ((-10, -20), (-30, -40), (-50, -60))
        tio.put_data(self.dfile, new_data, header="# More header\n", fmt='{:.2f}  ' * len(new_data), append='yes')

        res = tio.get_data(self.dfile, cols=[1,2])
        self.assertAlmostEqual(new_data[2][1], res[1][6], 3)

    def test_get_string(self):
        open(self.dfile, 'w').write("""# hello
one  two  three
four  five six

seven eight nine
""")
        res = tio.get_string(self.dfile)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[1][1], 'five')

        res = tio.get_string(self.dfile, cols=[1,2], nrows=2)
        self.assertEqual(len(res), 2)
        self.assertEqual(len(res[0]), 2)
        self.assertEqual(res[1][1], 'six')

        res = tio.get_string(self.dfile, cols=2, nrows=2)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[1], 'six')

        buf = open(self.dfile, 'r').readlines()

        res = tio.get_string(buf, cols=[1,2], nrows=2, buffer=True)
        self.assertEqual(len(res), 2)
        self.assertEqual(len(res[0]), 2)
        self.assertEqual(res[1][1], 'six')

    def test_rcols(self):
        open(self.dfile, 'w').write("""# hello
1  2  3
4 5 6

7 8 9
""")
        res = tio.rcols(self.dfile)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[1][1], 5)

        res = tio.rcols(self.dfile, cols=[1,2], nrows=2)
        self.assertEqual(len(res), 2)
        self.assertEqual(len(res[0]), 2)
        self.assertEqual(res[1][1], 6)

        res = tio.rcols(self.dfile, cols=2, nrows=2)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[1], 6)

    def test_get_datarray(self):
        open(self.dfile, 'w').write("""# hello
1  2  3
4 5 6

7 8 9
""")
        res = tio.get_datarray(self.dfile)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[1], 4)

        res = tio.get_datarray(self.dfile, cols=[1,2], nrows=2)
        self.assertEqual(len(res), 2)
        self.assertEqual(len(res[0]), 2)
        self.assertEqual(res[1][1], 6)

        res = tio.get_datarray(self.dfile, cols=2, nrows=2)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[1], 6)

        buf = open(self.dfile, 'r').readlines()

        res = tio.get_datarray(buf, cols=[1,2], nrows=2, buffer=True)
        self.assertEqual(len(res), 2)
        self.assertEqual(len(res[0]), 2)
        self.assertEqual(res[1][1], 6)

    def test_2Darray(self):
        data = np.array([[1,3,5],[7,9,11]])
        tio.put_2Darray(self.dfile, data, header="#Hello\n")

        res = tio.get_2Darray(self.dfile)
        self.assertEqual(res.shape, (2,3))
        self.assertEqual(res[0][2], 5.)

        tio.put_2Darray(self.dfile, np.array([[10,20,30]]), append='yes')
        res = tio.get_2Darray(self.dfile, cols=[1,2], nrows=3, verbose='yes')
        self.assertEqual(res.shape, (3,2))

class TestWCSUtil(unittest.TestCase):
    imgfile = os.path.join(ROOT, 'raw', 'test_raw.fits.fz')
    def test_arrscl(self):
        arr = np.array(range(100), dtype='float64')

        self.assertTrue(arr.min() < 10.)
        self.assertTrue(arr.max() > 20.)

        res = wcs.arrscl(arr, 10, 20)
        self.assertEqual(res.shape, (100,))
        self.assertFalse(res.min() < 10.)
        self.assertFalse(res.max() > 20.)

    def test_arrscl_alt(self):
        arr = np.array(range(1), dtype='float64')

        res = wcs.arrscl(arr, 10, 20, -10, 25)

        self.assertEqual(arr.shape, res.shape)
        self.assertEqual(arr[0], res[0])

    def test_arrscl_error(self):
        arr = np.array(range(25), dtype='float64')

        with capture_output() as (out, _):
            self.assertIsNone(wcs.arrscl(arr, 15, 99, 0, 0))
            output = out.getvalue().strip()
            self.assertTrue('arrmin must not' in output)

    def test_dict_get(self):
        d = {'key': 'value',
             'another': 'val'}

        self.assertEqual(wcs._dict_get(d, 'key'), d['key'])

        self.assertRaises(ValueError, wcs._dict_get, d, 'blah')

        self.assertEqual(wcs._dict_get(d, 'blah', 55), 55)

    def test_apply2dpolynomial(self):
        a = np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11]], dtype='float64')
        x = np.array([2,4,6,8], dtype='float64')
        y = np.array([3,6,9, 11], dtype='float64')

        v = wcs.Apply2DPolynomial(a, x, y)

        self.assertEqual(v.shape, (4,))
        self.assertEqual(v[1], 52542.)

    def test_make_xy_grid(self):
        numpts = 22
        x, y = wcs.make_xy_grid(numpts, [0, 10], [-5, 0])

        self.assertEqual(x.shape[0], numpts * numpts)
        self.assertEqual(y.shape[0], numpts * numpts)

        self.assertTrue(x.min() == 0)
        self.assertTrue(x.max() == 10.)
        self.assertTrue(y.min() == -5.)
        self.assertTrue(y.max() == 0)

        self.assertTrue(x[0] == 0.)
        self.assertTrue(x[-1] == 10.)
        self.assertTrue(y[0] == -5.)
        self.assertTrue(y[-1] == 0.)

        self.assertEqual(np.unique(x).shape[0], numpts)
        self.assertEqual(np.unique(y).shape[0], numpts)

    def test_make_amtrix(self):
        u = np.array(range(10))
        v = np.array(range(10))

        res = wcs.make_amatrix(u, v, 2)

        self.assertEqual(res.shape, (6,10))
        self.assertTrue(res.min() == 0.)
        self.assertTrue(res.max() == 81.)

        self.assertEqual(res[0][0], 1.)

        res = wcs.make_amatrix(u, v, 2, False)

        self.assertEqual(res.shape, (5,10))
        self.assertTrue(res.min() == 0.)
        self.assertTrue(res.max() == 81.)

        self.assertEqual(res[0][0], 0.)

        res = wcs.make_amatrix(u, v, 3)
        self.assertEqual(res.shape, (10,10))
        self.assertTrue(res.min() == 0.)
        self.assertTrue(res.max() == 729.)

    def test_invert_for_coeffs(self):
        u = np.random.random_sample((10,)) * 10.
        v = np.random.random_sample((10,)) * 10.

        x = np.random.random_sample((10,))
        y = np.random.random_sample((10,))
        am = wcs.make_amatrix(u, v, 2)

        xc, yc = wcs.invert_for_coeffs(am, x, y)

        self.assertTrue(xc.shape == yc.shape)

        xc1, yc1 = wcs.invert_for_coeffs(am, x, y, False)

        dx = abs(xc - xc1)
        dy = abs(yc - yc1)

        self.assertTrue(dx.max() < 0.00001)
        self.assertTrue(dy.max() < 0.00001)

    def test_pack_coeffs(self):
        u = np.random.random_sample((10,)) * 10.
        v = np.random.random_sample((10,)) * 10.

        x = np.random.random_sample((10,))
        y = np.random.random_sample((10,))
        am = wcs.make_amatrix(u, v, 2)

        xc, yc = wcs.invert_for_coeffs(am, x, y)

        xpc, ypc = wcs.pack_coeffs(xc, yc, 1)

        self.assertEqual(xpc.shape, (2,2))
        self.assertEqual(ypc.shape, (2,2))
        self.assertNotEqual(xpc[0][0], 0.)
        self.assertNotEqual(ypc[0][0], 0.)

        xpc, ypc = wcs.pack_coeffs(xc, yc, 1, False)

        self.assertEqual(xpc.shape, (2,2))
        self.assertEqual(ypc.shape, (2,2))
        self.assertEqual(xpc[0][0], 0.)
        self.assertEqual(ypc[0][0], 0.)

    def test_invert2dpolynomial(self):
        u = np.random.random_sample((10,)) * 10.
        v = np.random.random_sample((10,)) * 10.

        x = np.random.random_sample((10,))
        y = np.random.random_sample((10,))

        xc, yc = wcs.Invert2DPolynomial(u, v, x, y, 1)

        self.assertEqual(xc.shape, (2,2))
        self.assertEqual(yc.shape, (2,2))

        self.assertEqual(xc[1][1], 0.)

        xc, yc = wcs.Invert2DPolynomial(u, v, x, y, 2)

        self.assertEqual(xc.shape, (3,3))
        self.assertEqual(yc.shape, (3,3))

        self.assertEqual(xc[2][2], 0.)

        xc, yc = wcs.Invert2DPolynomial(u, v, x, y, 2, False, False)

        self.assertEqual(xc.shape, (5,))
        self.assertEqual(yc.shape, (5,))

    def test_ncoeff(self):
        self.assertEqual(wcs.Ncoeff(5), 21.0)

        self.assertEqual(wcs.Ncoeff(3), 10.)

        self.assertEqual(wcs.Ncoeff(3, False), 9.)


    def test_wcs_init(self):
        w = wcs.WCS(fits.open(self.imgfile)[1].header)

    def test_wcs_keys(self):
        wc = wcs.WCS(fits.open(self.imgfile)[1].header)
        self.assertEqual(102, len(wc.keys()))


    def test_image2sky_and_back(self):
        header = fits.open(self.imgfile)[1].header

        wc = wcs.WCS(header)

        xl = np.random.randint(3000, size=25)
        yl = np.random.randint(30, size=25)

        for x, y in zip(xl, yl):
            lat, long = wc.image2sky(x, y)
            xn, yn = wc.sky2image(lat, long)

            self.assertAlmostEqual(x, xn, 6)
            self.assertAlmostEqual(y, yn, 6)

        lat, long = wc.image2sky(xl, yl, False)
        xn, yn = wc.sky2image(lat, long, False, False)


        self.assertTrue(np.allclose(xl, xn, atol=1e-4))
        self.assertTrue(np.allclose(yl, yn, atol=1e-4))

        lat, long = wc.image2sky(xl, yl)
        xn, yn = wc.sky2image(lat, long, True, False)

        self.assertTrue(np.allclose(xl, xn, atol=1e-4))
        self.assertTrue(np.allclose(yl, yn, atol=1e-3))

        self.assertRaises(ValueError, wc.sky2image, [1, 2] ,[1])

    def test_extractprojection(self):
        header = fits.open(self.imgfile)[1].header

        wc = wcs.WCS(header)
        res = wc.ExtractProjection(wc)
        self.assertTrue('-TPV' in res)

        wc['ctype1'] = 'RA--AABB'
        self.assertRaises(ValueError, wc.ExtractProjection, wc)

    def test_applycdmatrix(self):
        x = np.random.random_sample((2,2)) * 10.
        y = np.random.random_sample((2,2)) * 10.
        header = fits.open(self.imgfile)[1].header

        wc = wcs.WCS(header)

        xc, yc = wc.ApplyCDMatrix(x, y)
        self.assertEqual(xc.shape, (2,2))
        self.assertEqual(yc.shape, (2,2))
        self.assertTrue(xc.min() > 0.)
        self.assertTrue(yc.max() < 0.)

        xxc, yyc = wc.ApplyCDMatrix(x, y, True)
        self.assertEqual(xc.shape, (2,2))
        self.assertEqual(yc.shape, (2,2))

    def test_image2sph_and_back(self):
        x = np.random.random_sample((10,)) * 100.
        y = np.random.random_sample((10,)) * 100.
        header = fits.open(self.imgfile)[1].header

        wc = wcs.WCS(header)

        lat, long = wc.image2sph(x, y)

        self.assertEqual(lat.shape, x.shape)
        self.assertEqual(long.shape, y.shape)
        self.assertTrue(0.0 <= lat.max() < 360.)
        self.assertTrue(-90.0 <= long.max() <= 90.0)
        self.assertTrue(0.0 <= lat.min() < 360.)
        self.assertTrue(-90.0 <= long.min() <= 90.0)

        xc, yc = wc.sph2image(lat, long)

        self.assertTrue(np.allclose(xc, x))
        self.assertTrue(np.allclose(yc, y))


        x = np.random.random_sample((10,)) * -100.
        y = np.random.random_sample((10,)) * -100.
        header = fits.open(self.imgfile)[1].header

        wc = wcs.WCS(header)

        lat, long = wc.image2sph(x, y)

        self.assertEqual(lat.shape, x.shape)
        self.assertEqual(long.shape, y.shape)
        self.assertTrue(0.0 <= lat.max() < 360.)
        self.assertTrue(-90.0 <= long.max() <= 90.0)
        self.assertTrue(0.0 <= lat.min() < 360.)
        self.assertTrue(-90.0 <= long.min() <= 90.0)

        xc, yc = wc.sph2image(lat, long)

        self.assertTrue(np.allclose(xc, x))
        self.assertTrue(np.allclose(yc, y))

        x = np.random.random_sample((9,)) * -100.
        y = np.random.random_sample((10,)) * -100.

        self.assertRaises(ValueError, wc.image2sph, x, y)

        lat = lat[4:]

        self.assertRaises(ValueError, wc.sph2image, lat, long)

        _ = wc.image2sph((np.random.random_sample((10,)) + 0.5) * 300, (np.random.random_sample((10,)) + 0.5) * 30)

        res = wc.sph2image([2, 5.], [40, 60.])

        self.assertEqual(res[0][0], 0.0)
        self.assertEqual(res[0][1], 0.0)
        self.assertEqual(res[1][0], 0.0)
        self.assertEqual(res[1][1], 0.0)

    def test_getpole(self):
        header = fits.open(self.imgfile)[1].header

        wc = wcs.WCS(header)

        a, b = wc.GetPole()

        self.assertAlmostEqual(6.0323610558, a, 8)
        self.assertAlmostEqual(-0.902896119, b, 8)

        wc.theta0 = 85.

        a, b = wc.GetPole()

        self.assertAlmostEqual(4.4615647290, a, 8)
        self.assertAlmostEqual(-0.815629657, b, 8)

        wc.theta0 = -85.

        a, b = wc.GetPole()

        self.assertAlmostEqual(7.603157382, a, 8)
        self.assertAlmostEqual(0.815629657, b, 8)

        wc.theta0 = 128.3

        self.assertRaises(ValueError, wc.GetPole)

        wc.theta0 = 333.5

        a, b = wc.GetPole()

        self.assertAlmostEqual(7.60315738265, a, 8)
        self.assertAlmostEqual(-0.2053879552, b, 8)

        wc.theta0 = 0.

        a, b = wc.GetPole()

        self.assertAlmostEqual(6.032361055, a, 8)
        self.assertAlmostEqual(0.667900207, b, 8)

    def test_getpole_corner_cases(self):
        header = fits.open(self.imgfile)[1].header

        wc = wcs.WCS(header)
        wc.theta0 = 0.

        wc.wcs['crval2'] = 0
        wc.longpole = 90.
        wc.latpole = 45.
        a, b = wc.GetPole()
        self.assertEqual(b, 45.0)

        wc.latpole = -89
        a, b = wc.GetPole()
        self.assertEqual(-89, b)

        wc = wcs.WCS(header)
        wc.wcs['crval1'] = 0.
        wc.wcs['crval2'] = 34.9
        wc.theta0 = 202.6
        a, b = wc.GetPole()
        self.assertAlmostEqual(a, math.pi / 2., 7)
        self.assertAlmostEqual(b, 0.567232, 5)

        wc.wcs['crval1'] = 0.
        wc.wcs['crval2'] = 5.4
        wc.theta0 = 5.4
        a, b = wc.GetPole()
        self.assertAlmostEqual(a, -2 * math.pi, 7)
        self.assertAlmostEqual(b, math.pi / 2., 7)

        wc.wcs['crval1'] = 0.
        wc.wcs['crval2'] = 5.4
        wc.theta0 = -5.4
        a, b = wc.GetPole()
        self.assertAlmostEqual(a, -math.pi, 7)
        self.assertAlmostEqual(b, -math.pi / 2.)

    def test_setangles(self):
        header = fits.open(self.imgfile)[1].header

        wc = wcs.WCS(header)

        wc.SetAngles(0, 90, 85)

        self.assertEqual(wc.longpole, 0)
        self.assertEqual(wc.latpole, 90)
        self.assertEqual(wc.theta0, 85)

        wc.wcs['longpole'] = 22.5
        wc.wcs['latpole'] = 18.325
        wc.wcs['theta0'] = 42.5

        wc.SetAngles(0, 90, 85)
        self.assertEqual(wc.longpole, 22.5)
        self.assertEqual(wc.latpole, 18.325)
        self.assertEqual(wc.theta0, 42.5)

    def test_distort(self):
        header = fits.open(self.imgfile)[1].header

        wc = wcs.WCS(header)

        x = np.random.random_sample((10,)) * 100.
        y = np.random.random_sample((10,)) * 100.

        xd, yd = wc.Distort(x, y)

        self.assertEqual(x.shape, (10,))

        y = np.random.random_sample((9,)) * 100.
        self.assertRaises(ValueError, wc.Distort, x, y)

        y = np.random.random_sample((10,)) * 100.
        wc.distort = None
        xd, yd = wc.Distort(x, y)

        self.assertTrue(np.array_equal(xd, x))
        self.assertTrue(np.array_equal(yd, y))


class TestAstrometry(unittest.TestCase):
    imgfile = os.path.join(ROOT, 'raw', 'test_raw.fits.fz')

    def test_circle_distance(self):
        self.assertEqual(ast.circle_distance(0, 0, 90, 0), 90.)
        self.assertAlmostEqual(5.0, ast.circle_distance(85, 0, 90, 0), 9)
        self.assertAlmostEqual(1.20139065, ast.circle_distance(22.5, -15, 22.56, -16.2), 8)
        self.assertAlmostEqual(0.51350694, ast.circle_distance(1.15, -.02, 1.65, 0.1, 'rad'), 8)

    def test_deg2dec_and_back(self):
        self.assertAlmostEqual(5.42083333, ast.deg2dec('05:25:15'))
        self.assertAlmostEqual(-5.42083333, ast.deg2dec('-05:25:15'))

        self.assertEqual(abs(ast.deg2dec('00:00:01.11')), abs(ast.deg2dec('-00:00:1.11')))

        self.assertAlmostEqual(5.42083333, ast.deg2dec(np.array(['05:25:15']))[0])
        self.assertAlmostEqual(-5.42083333, ast.deg2dec(np.array(['-05:25:15']))[0])

        val = '05:25:15'
        d = ast.deg2dec(val)
        dd = ast.dec2deg(d)

        self.assertEqual(val, dd[:len(val)])

        dd = ast.dec2deg(d, plussign=True)
        self.assertTrue(dd.startswith('+'))
        self.assertEqual(val, dd[1:len(val) + 1])

        dd = ast.dec2deg(d, short=True)
        self.assertEqual(val[:5], dd)

        dd = ast.dec2deg(d, short='ra')
        self.assertEqual(val[:6] + '02.5', dd)

        val = '-05:25:15'
        d = ast.deg2dec(val)
        dd = ast.dec2deg(d)

        self.assertEqual(val, dd[:len(val)])

        val = ['-05:25:15', '18:22:53.0']
        d = ast.deg2dec(val)

        dd = ast.dec2deg(d)

        self.assertEqual(dd[0][:len(val[0])], val[0])
        self.assertEqual(dd[1][:len(val[1])], val[1])

    def test_sky_area(self):
        ra = np.array([0, 0, 10, 10])
        dec = np.array([0, 10, 10, 0])

        res = ast.sky_area(ra, dec)
        self.assertTrue(res < 100.)
        self.assertTrue(100. - res < 0.3)

        dec = np.array([0, -10, -10, 0])
        res2 = ast.sky_area(ra, dec)
        self.assertAlmostEqual(res, res2, 9)

        ra = np.array([90, 90, 80, 80])
        res2 = ast.sky_area(ra, dec)
        self.assertAlmostEqual(res, res2, 9)

        ra = np.array([0, 0, 10, 10])
        dec = np.array([70, 80, 80, 70])

        res = ast.sky_area(ra, dec)
        self.assertTrue(res < 26.)
        self.assertTrue(26. - res < 0.3)

        res = ast.sky_area(ra, dec, units='rad')
        self.assertAlmostEqual(res, 0.007838561, 7)

    def test_get_pixelscale(self):
        header = fits.open(self.imgfile)[1].header
        r1 = ast.get_pixelscale(header, units='arcsec')
        r2 = ast.get_pixelscale(header, units='arcmin')
        r3 = ast.get_pixelscale(header, units='degree')

        self.assertRaises(ValueError, ast.get_pixelscale, header, units='rad')

        self.assertTrue(r1 > r2 > r3)

        self.assertAlmostEqual(r1 / 60., r2, 8)

        self.assertAlmostEqual(r2 / 60., r3, 8)

    def test_update_wcs_matrix(self):
        header = fits.open(self.imgfile)[1].header

        h2 = ast.update_wcs_matrix(header, 100, 100, 2160, 4146)

        self.assertAlmostEqual(h2['crval1'], 343.902416, 5)

class TestCCD_coreners(unittest.TestCase):
    imgfile = os.path.join(ROOT, 'raw', 'test_raw.fits')

    def test_desdm_corners(self):
        header = fits.open(self.imgfile)[1].header
        expected = [344.13095408267225,
                    -51.64229708325755,
                    343.89110998512797,
                    -51.559865443981,
                    343.88445326937233,
                    -51.71746333851118,
                    344.3716658821924,
                    -51.72437897595895,
                    344.37669586780265,
                    -51.566591752944554]

        corners = ccd.DESDM_corners(header)
        for i, val in enumerate(expected):
            self.assertAlmostEqual(val, corners[i], 8)

        corners2 = ccd.DESDM_corners(header, 200)
        for i, val in enumerate(corners):
            if i < 2:
                self.assertEqual(val, corners2[i])
            else:
                self.assertAlmostEqual(val, corners2[i], 1)

        header = fits.open(self.imgfile + '.fz')[1].header
        corners2 = ccd.DESDM_corners(header)
        for i, val in enumerate(corners):
            self.assertAlmostEqual(val, corners2[i], 8)

    def test_get_DESDM_corners_extant(self):
        r = np.array([0., 125., 175., 190., 250.])
        d = np.array([-25., 22, -86, 73., 43.6])
        res = ccd.get_DESDM_corners_extent(r, d)
        self.assertEqual(res[0], r[3])
        self.assertEqual(res[1], r[2])
        self.assertEqual(res[2], d.min())
        self.assertEqual(res[3], d.max())
        self.assertEqual(res[4], 'Y')

        r = np.array([125., 175., 190., 250.])
        d = np.array([22, -86, 73., 43.6])
        res = ccd.get_DESDM_corners_extent(r, d)
        self.assertEqual(res[0], r.min())
        self.assertEqual(res[1], r.max())
        self.assertEqual(res[2], d.min())
        self.assertEqual(res[3], d.max())
        self.assertEqual(res[4], 'N')

    def test_update_desdm_corners_messages_and_errors(self):
        logger = logging.getLogger('tester')
        with mock.patch.object(logger, 'info') as lgr:
            _ = ccd.update_DESDM_corners(None, logger=lgr)
            self.assertTrue(lgr.info.called)
            self.assertEqual(lgr.info.call_count, 2)

        with capture_output() as (out, _):
            _ = ccd.update_DESDM_corners(None, verb=True)
            output = out.getvalue().strip()
            self.assertTrue('Using header' in output)
            self.assertTrue('WARNING' in output)

        _, header = fitsio.read(self.imgfile, header=True)

        with mock.patch.object(logger, 'info') as lgr:
            _ = ccd.update_DESDM_corners(header, logger=lgr)
            self.assertTrue(lgr.info.called)
            self.assertEqual(lgr.info.call_count, 1)

        with capture_output() as (out, _):
            _ = ccd.update_DESDM_corners(header,get_extent=True, verb=True)
            output = out.getvalue().strip()
            self.assertTrue('Using header' in output)
            self.assertTrue('DECCMIN' in output)

        with capture_output() as (out, _):
            _ = ccd.update_DESDM_corners(header, verb=True)
            output = out.getvalue().strip()
            self.assertTrue('Using header' in output)
            self.assertFalse('DECCMIN' in output)

        with patch('despyastro.CCD_corners.get_DESDM_corners_extent', side_effect=Exception):
            with mock.patch.object(logger, 'info') as lgr:
                _ = ccd.update_DESDM_corners(header, get_extent=True, logger=lgr)
                self.assertTrue(lgr.info.called)
                self.assertEqual(lgr.info.call_count, 2)

        with patch('despyastro.CCD_corners.get_DESDM_corners_extent', side_effect=Exception):
            with capture_output() as (out, _):
                _ = ccd.update_DESDM_corners(header, get_extent=True)
                output = out.getvalue().strip()
                self.assertTrue('Problem computing' in output)

    def test_update_desdm_corners(self):
        _, header = fitsio.read(self.imgfile, header=True)
        h = ccd.update_DESDM_corners(header)

        for i in ['RA_CENT', 'DEC_CENT', 'RAC2', 'DECC2']:
            self.assertEqual(header[i], h[i])

        h = ccd.update_DESDM_corners(header, border=200)

        for i in ['RA_CENT', 'DEC_CENT', 'RAC2', 'DECC2']:
            self.assertAlmostEqual(header[i], h[i], 1)

        self.assertFalse('RACMIN' in header)

        h = ccd.update_DESDM_corners(header, get_extent=True)
        self.assertTrue('RACMIN' in h)


class TestZipper_interp(unittest.TestCase):
    size = 50
    xr = [0, 5, 22, 18, 49]
    yr = [range(5,40), range(3,12), range(8,40), range(8,40), range(10,22)]

    xc = [range(36,48), range(25,39), range(25,39)]
    yc = [36, 38, 4]

    xr1 = [0, 5, 22, 18, 49]
    yr1 = [range(0,25), range(3,12), range(8,40), range(8,40), range(10,22)]


    @classmethod
    def setUpClass(cls):
        cls.im = np.random.random_sample((cls.size, cls.size)) * 10.
        cls.c_msk = np.zeros((cls.size, cls.size), dtype=np.int32)
        cls.c_in_msk = np.zeros((cls.size, cls.size), dtype=np.int32)

        for i, v in enumerate(cls.yc):
            for j in cls.xc[i]:
                cls.c_msk[j, v] = 1
                if not i%2:
                    cls.c_in_msk[j, v] = 1
                else:
                    cls.c_in_msk[j, v] = 2

        cls.r_msk = np.zeros((cls.size, cls.size), dtype=np.int32)
        cls.r_in_msk = np.zeros((cls.size, cls.size), dtype=np.int32)

        for i, v in enumerate(cls.xr):
            for j in cls.yr[i]:
                cls.r_msk[v, j] = 1
                if not i%2:
                    cls.r_in_msk[v, j] = 1
                else:
                    cls.r_in_msk[v, j] = 2

    def test_zipper_interp(self):
        logger = logging.getLogger('tester')
        self.assertTrue(self.im.min() > 0.)
        in_vals = self.im[22][8:39]

        with mock.patch.object(logger, 'info') as lgr:
            im, ms = zpi.zipper_interp(self.im, self.r_msk, self.r_in_msk, 1, logger=lgr)

            dim = im-self.im
            self.assertTrue(dim.max() > 0.)

            out_vals = im[22][8:39]
            dif = abs(in_vals - out_vals)
            self.assertTrue(dif.min() > 0.)
            self.assertTrue(lgr.info.called)
            self.assertEqual(lgr.info.call_count, 1)


        self.assertTrue(self.im.min() > 0.)
        in_vals = self.im.transpose()[36][36:47]
        im, ms = zpi.zipper_interp(self.im, self.c_msk, self.c_in_msk, 2)

        dim = im-self.im
        self.assertTrue(dim.max() > 0.)

        out_vals = im.transpose()[36][36:47]

        dif = abs(in_vals - out_vals)
        self.assertTrue(dif.min() > 0.)

    def test_interp_errors(self):
        self.assertRaises(ValueError, zpi.zipper_interp, None, None, None, 5)

    def test_zipper_interp_rows(self):
        self.assertTrue(self.im.min() > 0.)
        in_vals = self.im[22][8:39]
        im, ms = zpi.zipper_interp_rows(self.im, self.r_msk, self.r_in_msk)

        dim = im-self.im
        self.assertTrue(dim.max() > 0.)

        out_vals = im[22][8:39]
        dif = abs(in_vals - out_vals)
        self.assertTrue(dif.min() > 0.)

        im, ms = zpi.zipper_interp_rows(self.im, self.r_msk, self.r_in_msk, DEFAULT_MAXCOLS=10)

        dim = im-self.im
        self.assertFalse(dim.max() > 0.)

        out_vals = im[22][8:39]
        dif = abs(in_vals - out_vals)
        self.assertFalse(dif.min() > 0.)

    def test_zipper_interp_left_right(self):
        img = np.random.random_sample((5, 5)) * 10.
        msk = np.array([[0, 1, 1, 2, 0],
                        [0, 0, 1, 1, 0],
                        [1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 0],
                        [1, 2, 1, 0, 1]], dtype=np.int32)
        in_msk = np.array([[1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 2, 2, 2],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1]], dtype=np.int32)
        inv_msk = np.array([3,3,3,2], dtype=np.int32)

        im, ms = zpi.zipper_interp_rows(img, msk, in_msk, invalid_mask=inv_msk)

        self.assertTrue(np.array_equal(ms, msk))
        self.assertFalse(np.array_equal(im, img))

        diff = img - im
        xd, yd = np.where(diff != 0)

        im1, ms1 = zpi.zipper_interp_rows(img, msk, in_msk, dilate=1, invalid_mask=inv_msk)

        diff = img - im1
        xd1, yd1 = np.where(diff != 0)
        self.assertTrue(len(xd) < len(xd1))
        self.assertTrue(len(yd) < len(yd1))

        im2 , ms2 = zpi.zipper_interp_rows(img, msk, in_msk, add_noise=True, invalid_mask=inv_msk)
        diff = img - im2
        xd2, yd2 = np.where(diff != 0)

        self.assertTrue(np.array_equal(xd, xd2))
        self.assertTrue(np.array_equal(yd, yd2))
        self.assertFalse(np.array_equal(im, im2))


        im3, ms3 = zpi.zipper_interp_rows(img, msk, in_msk, invalid_mask=inv_msk, BADPIX_INTERP=3)

        diff = img - im2
        xd3, yd3 = np.where(diff != 0)
        self.assertTrue(np.array_equal(xd, xd3))
        self.assertTrue(np.array_equal(yd, yd3))
        self.assertTrue(np.array_equal(im, im3))
        self.assertFalse(np.array_equal(ms, ms3))
        diff = ms - ms3
        mx, my = np.where(diff != 0)
        self.assertTrue(np.array_equal(mx, xd3))
        self.assertTrue(np.array_equal(my, yd3))


    def test_errors(self):
        with patch('despyastro.zipper_interp.np.where', side_effect=[[np.array([1,2,3]),np.array([5,6,7])],
                                                                              [np.array([9,8]), np.array([4,5])]]):
            with capture_output() as (out, _):
                res = zpi.zipper_interp_rows(self.im, self.r_msk, self.r_in_msk)
                output = out.getvalue().strip()
                self.assertEqual(1, res)
                self.assertTrue('Logic', output)

        with capture_output() as (out, _):
            _, _ = zpi.zipper_interp_cols(self.im, self.c_msk, self.c_in_msk, yblock=-2)
            output = out.getvalue().strip()
            self.assertTrue('yblock must' in output)

        logger = logging.getLogger('tester')
        with mock.patch.object(logger, 'info') as lgr:
            im, ms = zpi.zipper_interp_cols(self.im, self.r_msk, 2, ydilate=1, yblock=-2,
                                       region_file='file.reg',logger=lgr)
            self.assertTrue(os.path.exists('file.reg'))
            os.unlink('file.reg')

            self.assertTrue(lgr.info.called)
            self.assertEqual(lgr.info.call_count, 3)
        img = np.zeros((self.size, self.size), dtype=np.float32)
        with capture_output() as (out, _):
            _, ms = zpi.zipper_interp_cols(img, self.c_msk, self.c_in_msk, yblock=0)
            output = out.getvalue().strip()
            self.assertTrue('WARNING' in output and 'sampling' in output)

        logger = logging.getLogger('tester')
        with mock.patch.object(logger, 'info') as lgr:
            _, ms = zpi.zipper_interp_cols(img, self.c_msk, self.c_in_msk, yblock=0, logger=lgr)
            self.assertTrue(lgr.info.called)
            self.assertEqual(lgr.info.call_count, 3)

        with patch('despyastro.zipper_interp.np.all', side_effect=[False]):
            with capture_output() as (out, _):
                res = zpi.zipper_interp_cols(self.im, self.c_msk, self.c_in_msk, yblock=-2)
                self.assertEqual(res, 1)
                output = out.getvalue().strip()
                self.assertTrue('Logic' in output)

    def test_zipper_interp_cols(self):
        self.assertTrue(self.im.min() > 0.)
        in_vals = self.im.transpose()[36][36:47]
        im, ms = zpi.zipper_interp_cols(self.im, self.c_msk, self.c_in_msk)

        dim = im-self.im
        self.assertTrue(dim.max() > 0.)

        out_vals = im.transpose()[36][36:47]

        dif = abs(in_vals - out_vals)
        self.assertTrue(dif.min() > 0.)
        im1, ms = zpi.zipper_interp_cols(self.im, self.c_msk, 1, ydilate=1)
        diff = np.where(abs(im - im1) > 0.)
        self.assertTrue(len(diff[0]) > 0)

        im2, ms = zpi.zipper_interp_cols(self.im, self.c_msk, self.c_in_msk, DEFAULT_MAXCOLS=13)

        changes = np.where(im2 != im)
        self.assertTrue(changes[1].max() < 13)

        im2, ms = zpi.zipper_interp_cols(self.im, self.c_msk, self.c_in_msk, DEFAULT_MAXCOLS=25)

        changes = np.where(im2 != im)
        self.assertEqual(changes[1].shape, (0,))

        im2, ms = zpi.zipper_interp_cols(self.im, self.c_msk, self.c_in_msk, DEFAULT_MAXCOLS=2)

        changes = np.where(im2 != im)

        self.assertTrue(changes[1].min() > 2)
        self.assertTrue(changes[1].max() > 2)

        im2, ms = zpi.zipper_interp_cols(self.im, self.c_msk, self.c_in_msk, yblock=0)
        self.assertTrue(np.array_equal(np.where(im != self.im), np.where(im2 != self.im)))
        self.assertFalse(np.array_equal(im, im2))

        im1, ms1 = zpi.zipper_interp_cols(self.im, self.c_msk, self.c_in_msk, yblock=0, add_noise=True)
        diff = abs(im - im1)
        self.assertTrue(diff.max() > 0.)

        img = np.zeros((self.size, self.size), dtype=np.float32)

        print('aaa')
        im2, msk2 = zpi.zipper_interp_cols(self.im, self.c_msk, 2, ydilate=1)
        self.assertFalse(np.array_equal(np.where(im != self.im), np.where(im2 != self.im)))
        self.assertFalse(np.array_equal(im, im2))

        im4, msk4 = zpi.zipper_interp_cols(img, self.c_msk, self.c_in_msk, BADPIX_INTERP=8, region_file='file.reg')
        self.assertTrue(os.path.exists('file.reg'))
        os.unlink('file.reg')


if __name__ == '__main__':
    unittest.main()

