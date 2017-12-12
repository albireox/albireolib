#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Dec 11, 2017
# @Filename: misc.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import timeit


__all__ = ['timeit_func']


def timeit_func(func, *args, number=100, repeat=3, **kwargs):
    """Times a function.

    Wrapper around `timeit.repeat` that allows passing a function and
    the arguments with which it should be called.

    """

    func_name = func.__name__
    args_str = ','.join([str(repr(arg)) for arg in args]) + ',' if len(args) > 0 else ''
    kwargs_str = ','.join('{}={!r}'.format(key, kwargs[key]) for key in kwargs)

    tests = timeit.repeat(f'{func_name}({args_str} {kwargs_str})',
                          globals=func.__globals__,
                          repeat=repeat, number=number)

    return min(tests) / number
