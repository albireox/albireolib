#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Oct 14, 2017
# @Filename: base.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import urllib.request

__all__ = ['url_to_string']


def url_to_string(url):
    """Opens an URL and returns the contents as a string."""

    url_open = urllib.request.urlopen(url)

    return url_open.read().decode('utf-8')
