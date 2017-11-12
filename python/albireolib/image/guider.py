#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Nov 11, 2017
# @Filename: guider.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pathlib
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage.filters import laplace
from scipy.ndimage.morphology import binary_dilation

import astropy.io.fits as fits


def calculate_blur(image, mask=None, iterations=3):
    """Returns an estimate of the blur based on the Laplacian variance.

    Provides an estimate of how blurry an image is by applying an Laplace
    filter and then calculating the variance. A ``mask`` can be passed to
    be applied to data *after* the Laplace filter has been used but before
    calculating the variance. If ``iterations`` is defined, a
    `~scipy.ndimage.binary_dilation` is applied to the mask ``iteration``
    times. This is useful to remove the edges of the image before estimating
    the blurriness.

    """

    assert isinstance(image, np.ndarray), 'image input must be an array'
    assert image.ndim == 2, 'invalid number of dimensions'

    if mask is not None and iterations is not None:
        dilated_mask = binary_dilation(mask, iterations=iterations)
        lap = laplace(image)
        return np.ma.var(np.ma.array(lap, mask=dilated_mask))
    else:
        return laplace(image).var()


def plot_gimage_blur(files, plot_order_by='cart', extension=2, mask_extension=3):
    """Returns a plot of blur measurements for a list of images.

    Parameters:
        files (str, `~pathlib.Path`, or list):
            A list of filepaths as strings or `~pathlib.Path` objects.
        plot_order_by (str):
            The order in which the files are plotted. Can be ``None``, in
            which case the input order will be user, or ``'cart'`` for sorting
            the input files by cart id.
        extension (int):
            The extension in the input file that contains the data to plot.
        mask_extension (int):
            The extension in the input file that contains the mask for the
            data.

    Returns:
        plot_data (tuple):
            A tuple in which the first element is the plot
            `~matplotlib.figure.Figure`, the second are the
            `~matplotlib.axes.Axes`, the last one is a list of blur
            measurements as returned by `.calculate_blur`. The list of blur
            estimates has the same order as the input ``files``. Some of the
            elements may be `numpy.nan` if the blurriness cannot be calculated.

    """

    plt.style.use(['seaborn-deep', 'seaborn-white'])

    if isinstance(files, (str, pathlib.Path)):
        files = [files]

    plot_data = []

    carts = []
    gimgs = []

    # We want the returned list of blur measurements to have the same size as
    # the input, but to plot only the good values.
    blur_meas_plot = []
    blur_meas_return = []

    for fn in files:

        fn_path = pathlib.Path(fn)
        data = fits.getdata(fn_path, extension)
        mask = fits.getdata(fn_path, mask_extension) > 0

        if data.shape[0] < 100 or data.shape[1] < 10:
            warnings.warn(f'invalid image {fn!s} with shape {data.shape}.')
            blur_meas_return.append(np.nan)
            continue

        frame = int(fits.getheader(fn, 0)['OBJECT'].split('-')[1])
        flat_files = list(fn_path.parent.glob(f'flat-{frame:04d}-*.dat'))

        if len(flat_files) == 0 or len(flat_files) > 1:
            warnings.warn(f'no flat found for image {fn!s}.')
            blur_meas_return.append(np.nan)
            continue

        cart = int(flat_files[0].name.split('-')[2].split('.')[0])

        plot_data.append(np.ma.array(data, mask=mask))
        carts.append(cart)
        gimgs.append(fits.getheader(fn, 0)['OBJECT'])

        blur_meas_plot.append(calculate_blur(data, mask))
        blur_meas_return.append(calculate_blur(data, mask))

    plot_order = np.arange(len(plot_data))
    if plot_order_by == 'cart':
        plot_order = np.argsort(carts)

    n_pix_height, n_pix_width = np.max(list(zip(*[im.shape for im in plot_data])), axis=1)
    plot_image = np.ma.zeros((n_pix_height, n_pix_width * len(plot_data))) + np.nan

    for ii, order_idx in enumerate(plot_order):
        im = plot_data[order_idx]
        im_height, im_width = im.shape
        plot_image[0:im_height, n_pix_width * ii:n_pix_width * ii + im_width] = im
        plot_image[0:im_height, n_pix_width * ii:n_pix_width * ii + im_width].mask = im.mask

    cmap = matplotlib.cm.viridis
    cmap.set_bad('w', 1.)

    fig = plt.figure(figsize=(plot_image.shape[1] / 50., plot_image.shape[0] / 50. + 1), dpi=150)
    ax = fig.add_subplot(111)

    ax.imshow(plot_image, cmap=cmap, aspect=1, origin='lower', interpolation='none')

    for ii, order_idx in enumerate(plot_order):

        cart = carts[order_idx]
        gimg = gimgs[order_idx]
        blur = blur_meas_plot[order_idx]

        ax.text(n_pix_width * ii + n_pix_width / 2., n_pix_height + 5,
                f'#{cart}: {gimg}', rotation=90, va='bottom', ha='center', fontsize=11)

        ax.text(n_pix_width * ii + n_pix_width / 2., -5,
                f'{blur:.6f}', rotation=90, va='top', ha='center', fontsize=11)

    plt.axis('off')

    return fig, ax, blur_meas_return
