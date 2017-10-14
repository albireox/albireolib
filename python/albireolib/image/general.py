#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Oct 12, 2017
# @Filename: general.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import itertools
import re
import warnings

import astropy.convolution
import astropy.modeling
import astropy.wcs

import numpy as np


__all__ = ['crop_hdu', 'replace_wcs', 'fwhm_to_sigma', 'sigma_to_fwhm',
           'gaussian_kernel_from_fwhm', 'gaussian_filter', 'fit_gaussian']


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

    new_hdu.data = np.array(data)

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

    Parameters:
        hdu (~astropy.io.fits.ImageHDU):
            The `~astropy.io.fits.ImageHDU` whose WCS definition we will
            replace.
        wcs (~astropy.wcs.WCS):
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

    return fwhm / 2 / np.sqrt(2 * np.log(2))


def sigma_to_fwhm(sigma):
    """Returns the FWHM for a sigma."""

    return sigma * 2 * np.sqrt(2 * np.log(2))


def gaussian_kernel_from_fwhm(fwhm, pixel_scale=1, **kwargs):
    """Returns a Gaussian kernel for a FWHM.

    Parameters:
        fwhm (float):
            The FWHM (seeing) of the Gaussian kernel, in arcsec.
        pixel_scale (float):
            The pixels scale, in arcsec.
        kwargs (dict):
            Other parameters to be passed to
            `~astropy.convolution.Gaussian2DKernel`.

    Returns (`~astropy.convolution.Gaussian2DKernel`):
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
    xmean, ymean = np.array(shape) / 2.

    xx, yy = np.mgrid[:shape[0], :shape[1]]

    g_init = astropy.modeling.models.Gaussian2D(amplitude=1., x_mean=xmean, y_mean=ymean,
                                                x_stddev=1., y_stddev=1.)

    f2 = astropy.modeling.fitting.LevMarLSQFitter()

    gg = f2(g_init, xx, yy, array)

    return gg
