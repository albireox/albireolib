#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Oct 14, 2017
# @Filename: ps1.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import shutil

from ..utils.base import url_to_string

import astropy.utils.data
import astropy.table


__all__ = ['Downloader']


class Downloader(object):
    """A wrapper around PS1's ``ps1filenames.py``.

    Provides a wrapper around the PS1 image list service (see
    `documentation <https://confluence.stsci.edu/display/PANSTARRS/PS1+Image+Cutout+Service#PS1ImageCutoutService-ImageListService>`_).
    Upon instantiation, queries ``ps1filenames.py`` to determine the sky cell
    that contains the input coordinates and retrieves a list of images.

    Parameters:
        ra (float):
            The right ascension of the sky cell to download.
        dec (float):
            The declination of the sky cell to download.
        filters (str):
            A string of filters to download, e.g. ``'griz'``.
        types (list):
            A list of image types to downalod. Available types include
            ``stack, stack.wt, stack.mask, stack.exp, stack.num, stack.expwt,
            stack.psf, stack.mdc, and stack.cmf``.

    Attributes:
        images (`~astropy.table.Table`):
            A table with the list of images returned by ``ps1filenames.py``
            for the input parameters.

    """

    BASE_URL = 'http://ps1images.stsci.edu'

    def __init__(self, ra, dec, filters='griz',
                 types=['stack', 'stack.wt', 'stack.psf', 'stack.mask']):

        url = self.BASE_URL + '/cgi-bin/ps1filenames.py' + \
            f'?ra={ra:.6f}&dec={dec:.6f}&filters={filters}' + '&type={}'.format(','.join(types))

        self.images = astropy.table.Table.read(url_to_string(url), format='ascii')

    def download(self, path='./'):
        """Downloads all the images to a path."""

        cwd = os.getcwd()

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        os.chdir(path)

        urls = [self.BASE_URL + url for url in self.images['filename']]
        files = astropy.utils.data.download_files_in_parallel(urls)

        for ii, fn in enumerate(files):
            shutil.move(fn, self.images['shortname'][ii])

        os.chdir(cwd)
