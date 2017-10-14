#!/usr/bin/env python
# encoding: utf-8
#
# dss.py
#
# Created by José Sánchez-Gallego on 14 Sep 2017.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from astropy.io import fits

import io
import pathlib
import urllib.request

import numpy as np

try:
    import PIL.Image
except ImportError:
    raise ImportError('this module requires pillow.')


__all__ = ['download_dss', 'DSS']


BASE_URL = ('http://stdatu.stsci.edu/cgi-bin/dss_search?'
            'v={survey_full}&r={ra:.6f}&d={dec:.6f}&e=J2000&h={height}&'
            'w={width}&f={format}&c=none&fov=NONE&v3=')

survey_full_dict = {'1r': 'poss1_red',
                    '1b': 'poss1_blue',
                    '2r': 'poss2ukstu_red',
                    '2b': 'poss2ukstu_blue',
                    '2ir': 'poss2ukstu_ir'}


class DSS(fits.HDUList):
    """Creates a DSS FITS object around a region of the sky.

    Parameters:
        ra,dec (float):
            Right Ascension and Declination around of the DSS region that will
            be created.
        size (float or list):
            The size, in arcmin, of the area to download. It can be specified
            as a single value, in which case a square region will be
            downloaded, or a list of two values, height and width.
        survey ({'2r', '2b', '2ir', '1r', '1b'}):
            The version of the survey to use for downloads.

    """

    def __init__(self, ra, dec, size=5, survey='2r', **kwargs):

        size = np.atleast_1d(size)
        assert len(size) > 0 and len(size) <= 2, 'incorrect shape for size.'

        assert ra >= 0 and ra < 360., 'invalid RA coordinate'
        assert dec > -90 and dec < 90, 'invalid Dec coordinate'

        if len(size) == 1:
            height = width = size[0]
        else:
            height, width = size

        assert survey in ['2r', '2ir', '2b', '1r', '1b'], 'invalid survey type.'

        survey_full = survey_full_dict[survey]

        self.url = BASE_URL.format(survey_full=survey_full, ra=ra, dec=dec,
                                   height=height, width=width, format='fits')

        try:
            url_data = urllib.request.urlopen(self.url)
        except urllib.request.URLError as ee:
            raise ValueError(f'cannot open URL for these parameters: {ee}')

        data = url_data.read()

        if 'Something went wrong' in str(data):
            if 'Calibration and image data not available for field data...' not in str(data):
                raise ValueError(f'survey {survey_full} does not cover coordiantes ({ra},{dec})')
            else:
                raise ValueError('unknown problem while retrieving the data.')

        fits_obj = fits.HDUList.fromstring(data)

        super(DSS, self).__init__(fits_obj)

        self._gif = None

    def writeto(self, *args, **kwargs):
        """Writes the FITS. See `~astropy.io.fits.HDUList.writeto`."""

        new_obj = fits.HDUList([ext for ext in self])

        return new_obj.writeto(*args, **kwargs)

    @property
    def gif(self):
        """Returns a PILImage_ object for this field.

        .. _PILImage: http://pillow.readthedocs.io/en/latest/reference/Image.html

        """

        if self._gif is None:

            url = self.url.replace('f=fits', 'f=gif')

            try:
                url_data = urllib.request.urlopen(url)
            except urllib.request.URLError as ee:
                raise ValueError(f'cannot open URL for these parameters: {ee}')

            data = url_data.read()

            image = PIL.Image.open(io.BytesIO(data))
            assert isinstance(image, PIL.GifImagePlugin.GifImageFile), 'incorrect image type.'

            self._gif = image

        return self._gif


def download_dss(ra, dec, path, save_gif=False, overwrite=False, **kwargs):
    """Downloads a DSS image.

    Parameters:
        ra,dec (float):
            Right Ascension and Declination of the DSS region to be downloaded.
        path (str):
            The full path where the FITS image will be saved.
        save_gif (bool):
            If ``True``, the GIF image will also be saved to the same location
            defined in ``path``.
        overwrite (bool):
            Whether the images, if they exist, should be overwritten.
        kwargs (dict):
            Other arguments to be passed to :class:`DSS`.

    Return:
        DSS object:
            A :class:`DSS` object representing the field.

    """

    dss = DSS(ra, dec, **kwargs)

    path = pathlib.Path(path)

    assert not path.is_dir(), 'path must contain a filename.'
    assert path.parent.exists(), 'directory does not exist.'

    dss.writeto(path, overwrite=overwrite)

    if save_gif:
        gif = dss.gif
        gif_path = path.with_suffix('.gif')
        if gif_path.exists():
            if overwrite:
                gif_path.unlink()
            else:
                raise FileExistsError(f'file {gif_path:s} exists and overwrite=False.')
        gif.save(gif_path)

    return dss
