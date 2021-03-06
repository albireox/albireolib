#!/usr/bin/env python
# encoding: utf-8
#
# setup.py
#


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from setuptools import setup, find_packages

import os


requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
install_requires = [line.strip().replace('==', '>=') for line in open(requirements_file)
                    if not line.strip().startswith('#') and line.strip() != '']

NAME = 'albireolib'
VERSION = '0.1.0'
RELEASE = 'dev' in VERSION


def run(packages):

    setup(name=NAME,
          version=VERSION,
          license='BSD3',
          description='Description of your project.',
          long_description=open('README.rst').read(),
          author='José Sánchez-Gallego',
          author_email='gallegoj@uw.edu',
          keywords='astronomy software',
          url='https://github.com/albireox/albireolib',
          include_package_data=True,
          packages=packages,
          # install_requires=install_requires,
          package_dir={'': 'python'},
          scripts=[],
          classifiers=[
              'Development Status :: 4 - Beta',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: BSD License',
              'Natural Language :: English',
              'Operating System :: OS Independent',
              'Programming Language :: Python',
              'Programming Language :: Python :: 2.6',
              'Programming Language :: Python :: 2.7',
              'Topic :: Documentation :: Sphinx',
              'Topic :: Software Development :: Libraries :: Python Modules',
          ],
          )


if __name__ == '__main__':

    packages = find_packages(where='python')

    # Runs distutils
    run(packages)
