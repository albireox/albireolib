#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2018-07-12
# @Filename: monkeypatch.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-07-19 19:15:38

import astropy

from ..misc.color_print import color_text


def hdulist_repr(self):
    """Custom representation for `astropy.io.fits.HDUList`."""

    table = astropy.table.Table(None, names=['No.', 'Name', 'Ver', 'Type', 'Cards',
                                             'Dimensions', 'Format'],
                                dtype=[int, 'S100', int, 'S20', int, tuple, 'S10'])

    default = ('', 1, '', 0, (), '')

    for ext_no, ext in enumerate(self):

        summary = list(ext._summary()[:-1])

        if len(summary) < len(default):
            summary += default[len(summary):]

        summary = [ext_no] + summary
        summary[5] = summary[5][::-1]

        table.add_row(summary)

    ptable = table.pformat(max_lines=-1, max_width=-1)
    ptable[0] = color_text(ptable[0], 'red')
    ptable[1] = color_text(ptable[1], 'red')

    my_repr = 'Filename: {}\n'.format(self.filename())
    my_repr += '\n'.join(ptable)

    return my_repr


def hdu_repr(self):
    """Custom representation for `astropy.io.fits.hdu._BaseHDU`."""

    rep = ''

    if hasattr(self, 'fileinfo'):
        rep += 'Filename: {}\n'.format(self.fileinfo()['file'].name)

    rep += 'Type: {}\n'.format(self.__class__.__name__)

    if hasattr(self, '_summary'):
        summary = self._summary()

        rep += 'Extname: {}\n'.format(summary[0])
        rep += 'Image info:\n'
        rep += '   data type: {}\n'.format(summary[5])
        rep += '   dims: {}\n'.format(summary[4][::-1])

    if len(rep) == 0:
        return super(self.__class__, self).__repr__()

    return rep


def hdu_getitem(self, sl):

    return self.data.__getitem__(sl)


def monkeypatch_astropy_fits(module=None):
    """Overrides methods in ``astropy.io.fits`` with custom versions."""

    if module is None:
        import astropy
        module = astropy.io.fits

    module.HDUList.__repr__ = hdulist_repr
    module.hdu.base._BaseHDU.__repr__ = hdu_repr

    module.hdu.ImageHDU.__getitem__ = hdu_getitem
    module.hdu.PrimaryHDU.__getitem__ = hdu_getitem
    module.hdu.CompImageHDU.__getitem__ = hdu_getitem
