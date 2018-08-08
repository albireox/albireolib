#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2017-10-012
# @Filename: general.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-08-08 16:03:51


from __future__ import absolute_import, division, print_function

import itertools
import re
import warnings

import astropy.convolution
import astropy.modeling
import astropy.modeling.models
import astropy.wcs
import numpy


__all__ = ['crop_hdu', 'replace_wcs', 'fwhm_to_sigma', 'sigma_to_fwhm',
           'gaussian_kernel_from_fwhm', 'gaussian_filter', 'fit_gaussian',
           'CCD', 'SyntheticImage']


def crop_hdu(hdu, xmin, xmax, ymin, ymax, return_wcs=True, ignore_warnings=True):
    """Crops a HDU.

    Trims the input HDU and, optionally, calculates the WCS of the resulting
    image. It returns a new `~astropy.io.fits.ImageHDU` object with the cropped
    image. If ``return_wcs=True`` (the default), also returns the WCS
    definition for the cropped image.

    Parameters:
        hdu (~astropy.io.fits.ImageHDU):
            The `~astropy.io.fits.ImageHDU` that will be cropped.
        xmin,xmax,ymin,ymax (int):
            The section of ``hdu`` to crop.
        return_wcs (bool):
            If *True*, and the input HDU contains WCS information, will return
            a `~astropy.wcs.WCS` object with the WCS definition for the cropped
            image. Returns ``None`` if the input HDU does not contain WCS
            information.
        ignore_warnings (bool):
            If *True*, warnings raised during the creation of the WCS object
            will be silenced.

    """

    new_hdu = hdu.copy()
    hdu_shape = new_hdu.data.shape

    assert xmin > 0 and ymin > 0 and xmax < hdu_shape[1] and ymax < hdu_shape[0], \
        'invalid crop region.'

    data = new_hdu.data.copy()
    data = data[ymin:ymax, xmin:xmax]

    new_hdu.data = numpy.array(data)

    if return_wcs is False:
        return new_hdu

    with warnings.catch_warnings():

        if ignore_warnings:
            warnings.simplefilter('ignore', astropy.wcs.FITSFixedWarning)

        # Checks whether this image has WCS information
        wcs_list = astropy.wcs.find_all_wcs(new_hdu.header)
        if len(wcs_list) == 0:
            return new_hdu, None
        else:
            new_wcs = wcs_list[0].deepcopy()

    new_wcs.wcs.crpix[0] -= xmin
    new_wcs.wcs.crpix[1] -= ymin

    return new_hdu, new_wcs


def replace_wcs(hdu, wcs):
    """Replaces WCS information in the header.

    Removes the current WCS information in the input
    `~astropy.io.fits.ImageHDU` and replace it with new one.

    Parameters
    ----------
    hdu : `~astropy.io.fits.ImageHDU`:
        The `~astropy.io.fits.ImageHDU` whose WCS definition we will
        replace.
    wcs : `~astropy.wcs.WCS`:
        A `~astropy.wcs.WCS` object containing the new WCS definition.

    """

    # Checks for old WCS keys in the form PC001002
    pc_old_pattern = re.compile('PC0*[0-9]{1}0*[0-9]{1}')
    header_keys = hdu.header.keys()
    pc_old_in_header = filter(pc_old_pattern.match, header_keys)

    wcs_keys = wcs.to_header().keys()

    for key in itertools.chain(wcs_keys, pc_old_in_header):
        if key in hdu.header:
            del hdu.header[key]

    # Adds the new WCS header to the hdu
    hdu.header.extend(wcs.to_header().cards)

    return hdu


def fwhm_to_sigma(fwhm):
    """Returns the sigma for a FWHM."""

    return fwhm / 2 / numpy.sqrt(2 * numpy.log(2))


def sigma_to_fwhm(sigma):
    """Returns the FWHM for a sigma."""

    return sigma * 2 * numpy.sqrt(2 * numpy.log(2))


def gaussian_kernel_from_fwhm(fwhm, pixel_scale=1, **kwargs):
    """Returns a Gaussian kernel for a FWHM.

    Parameters
    ----------
    fwhm : `float`
        The FWHM (seeing) of the Gaussian kernel, in arcsec.
    pixel_scale : `float`
        The pixels scale, in arcsec.
    kwargs : `dict`
        Other parameters to be passed to
        `~astropy.convolution.Gaussian2DKernel`.

    Returns
    -------
    kernel : `~astropy.convolution.Gaussian2DKernel`
        An astropy `~astropy.convolution.Gaussian2DKernel` kernel for the
        input FHWM.

    """

    stddev = fwhm_to_sigma(fwhm) / pixel_scale

    return astropy.convolution.Gaussian2DKernel(stddev, **kwargs)


def gaussian_filter(stddev, array):
    """Convolves an array with a Gaussian filter."""

    return astropy.convolution.convolve(
        array, astropy.convolution.Gaussian2DKernel(stddev))


def fit_gaussian(array):
    """Fits a 2D gaussian to an array of data."""

    shape = array.shape
    xmean, ymean = numpy.array(shape) / 2.

    xx, yy = numpy.mgrid[:shape[0], :shape[1]]

    g_init = astropy.modeling.models.Gaussian2D(amplitude=1., x_mean=xmean, y_mean=ymean,
                                                x_stddev=1., y_stddev=1.)

    f2 = astropy.modeling.fitting.LevMarLSQFitter()

    gg = f2(g_init, xx, yy, array)

    return gg


class CCD(object):
    """A class representing the parameters that define a CCD chip.

    Parameters
    ----------
    shape : `tuple`
        The shape of the image to generate.
    pixel_size : `float`
        The pixel size, in microns. Assumes the pixel is square.
    read_noise : `float`
        The RMS of the read noise, in electrons.
    gain : `float`
        The gain in electrons per ADU.
    name : `str` or None
        A string with the name of the CCD chip (e.g., its model or SN).

    """

    def __init__(self, shape, pixel_size, read_noise=1.0, gain=1.0, name=None):

        self.shape = shape
        self.pixel_size = pixel_size
        self.read_noise = read_noise
        self.gain = gain
        self.name = name


class SyntheticImage(object):
    """Creates and image with Gaussian features, bias, and noise.

    Parameters
    ----------
    ccd : .CCD
        A `.CCD` object describing the chip that produces this image.
    xy : list
        A list of tuples in which each tuple are the ``(x, y)`` coordinates of
        the Gaussian sources.
    sigma_x : list or float
        A list of floats with the same length as ``xy`` in which each element
        is the x-axis sigma for the Gaussian sources. Alternatively, a single
        float which will be applied to all the sources.
    sigma_y : list or float
        As ``sigma_x`` but for the y axis.
    fluxes : list or float
        The total flux of the Gaussian sources. Same format as ``sigma_x``.
    peaks : list or float
        The peak of each of the Gaussian sources. Same format as ``sigma_x``.
        Cannot be defined at the same time as ``fluxes``.
    bias : float or None
        The bias level of the image. If ``None``, no bias level will be added.
    cosmic_p : float
        The p factor for the binomial distribution used to model cosmic rays.
    exp_time : float
        The exposure time, in seconds. Used as a multiplicative factor for the
        log-normal distribution to estimate the total dark current.
    dark_sigma : float
        The sigma of the log-normal dark current distribution.
    sample_box : int
        The length of the box used to sample the Gaussian source.

    Attributes
    ----------
    signal : `numpy.ndarray`
        The array representing the image signal.
    noise : `numpy.ndarray`
        The array representing the image noise.
    sources : `list`
        A list of `~astropy.modeling.functional_models.Gaussian2D` objects that
        have been added to the image.

    """

    def __init__(self, ccd, xy=None, sigma_x=None, sigma_y=None,
                 fluxes=None, peaks=None, bias=400., read_noise=1.,
                 cosmic_p=0.005, exp_time=1., dark_sigma=5., sample_box=100):

        self.ccd = ccd
        self.signal = numpy.zeros(self.ccd.shape[::-1], dtype=numpy.float32)
        self.noise = numpy.zeros(self.ccd.shape[::-1], dtype=numpy.float32)

        # Add a bias level
        self.bias = 0.0
        if bias is not None:
            self.add_bias_level(bias)

        self.sample_box = sample_box
        assert self.sample_box % 2 == 0, 'sample_box must be even.'

        # We use a small grid to make the computing of the source Gaussian faster
        # and define the meshgrid here to avoid having to repeat this for each source
        self._meshgrid = numpy.mgrid[0:sample_box, 0:sample_box]

        # Add sources
        self.sources = []

        if xy is not None:

            assert sigma_x is not None, 'sigma_x cannot be None'

            sigma_x = numpy.atleast_1d(sigma_x)
            sigma_x = numpy.tile(sigma_x, len(xy)) if len(sigma_x) == 1 else sigma_x

            if sigma_y is not None:
                sigma_y = numpy.atleast_1d(sigma_y)
                sigma_y = numpy.tile(sigma_y, len(xy)) if len(sigma_y) == 1 else sigma_y
            else:
                sigma_y = sigma_x

            assert sigma_y is None or len(sigma_x) == len(sigma_y), \
                'invalid length for sigma_x or sigma_y'

            assert fluxes is not None or peaks is not None, \
                'either fluxes or peaks need to be defined'

            if peaks is not None:
                peaks = numpy.atleast_1d(peaks)
                peaks = numpy.tile(peaks, len(xy)) if len(peaks) == 1 else peaks
                assert fluxes is None, 'fluxes cannot be active at the same time as peaks'

            if fluxes is not None:
                fluxes = numpy.atleast_1d(fluxes)
                fluxes = numpy.tile(fluxes, len(xy)) if len(fluxes) == 1 else fluxes

                assert peaks is None, 'peaks cannot be active at the same time as fluxes'

                # Convert to peaks
                peaks = fluxes / (2. * numpy.pi * sigma_x * sigma_y)

            for ii in range(len(xy)):
                self.add_source(xy[ii], peaks[ii], sigma_x[ii], sigma_y[ii])

        if self.ccd.read_noise is not None:
            self.noise += self.get_read_noise()

    @property
    def image(self):
        """Returns the signal plus its associated noise."""

        return self.signal + self.noise

    @property
    def snr(self):
        """Returns the signal to noise ratio for each element in the image."""

        return self.signal / self.noise

    def get_read_noise(self):
        """Returns an array of read noise assuming a normal distribution."""

        read_noise_adu = self.ccd.read_noise / self.ccd.gain
        return numpy.random.normal(scale=read_noise_adu, size=self.image.shape)

    def add_bias_level(self, bias):
        """Adds a bias level to the image.

        A ``read_noise`` noise is added to the ``bias`` value. The attribute
        `~bias` is updated with the bias level added by this method.

        Parameters
        ----------
        bias : float
            The bias level to add.

        """

        self.bias += bias
        self.signal += bias

    def add_source(self, xy, peak, sigma_x, sigma_y):
        """Adds a series of Gaussian sources with noise to the image.

        Parameters
        ----------
        xy : tuple
            A tuple containing the ``(x, y)`` coordinates of the source.
        peak : float
            The height of the peak of the Gaussian.
        sigma_x : float
            The sigma across the x axis.
        sigma_y : float
            The sigma across the y axis.

        Returns
        -------
        out : `~astropy.modeling.functional_models.Gaussian2D`
            The astropy `~astropy.modeling.functional_models.Gaussian2D` object
            used to create this source. Poisson noise will be added.

        """

        assert len(xy) == 2, 'invalid lenght for xy'

        xx, yy = xy

        x_offset = int(xx) - int(self.sample_box / 2) if xx > self.sample_box / 2 else 0
        y_offset = int(yy) - int(self.sample_box / 2) if yy > self.sample_box / 2 else 0

        x_box = xx - x_offset
        y_box = yy - y_offset

        gaussian = astropy.modeling.models.Gaussian2D(peak, x_box, y_box, sigma_x, sigma_y)
        y_grid, x_grid = self._meshgrid

        gauss_data = gaussian(x_grid, y_grid)

        poisson_noise = numpy.random.poisson(gauss_data)
        noise = gauss_data - poisson_noise

        self.signal[y_offset:y_offset + self.sample_box,
                    x_offset:x_offset + self.sample_box] += gauss_data

        self.noise[y_offset:y_offset + self.sample_box,
                   x_offset:x_offset + self.sample_box] += noise

        gaussian.x_mean += x_offset
        gaussian.y_mean += y_offset
        self.sources.append(gaussian)

        return gaussian
