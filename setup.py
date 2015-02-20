#!/usr/bin/env python
# setup.py build file

DESCRIPTION = "BinOcular Retinal Image Statistics"
LONG_DESCRIPTION = """\
Berkeley BORIS is a project to analyze the statistics of the visual world
during natural viewing.
"""

NAME = "boris"
AUTHOR = "Bill Sprague and Emily Cooper"
AUTHOR_EMAIL = "bill.sprague@berkeley.edu"
DOWNLOAD_URL = "https://github.com/Berkeley-BORIS/BORIS_Code.git"
LICENSE = "MIT"
VERSION = "0.1.dev2"

from setuptools import setup, find_packages

if __name__ == '__main__':
    setup(name=NAME,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        download_url=DOWNLOAD_URL,
        version=VERSION,
        packages=find_packages(exclude='boris.cli'),
        classifiers=[
                     'Development Status :: 1 - Planning',
                     'Intended Audience :: Science/Research',
                     'Programming Language :: Python :: 2'
                     'Programming Language :: Python :: 2.7',
                     #'Programming Language :: Python :: 3.3',
                     #'Programming Language :: Python :: 3.4',
                     'License :: OSI Approved :: MIT License',
                     'Topic :: Scientific/Engineering',
                     'Operating System :: POSIX',
                     'Operating System :: Unix',
                     'Operating System :: MacOS'],
        install_requires=['numpy', 'pandas', 'Click', 'pytables', 'pyyaml'],
        entry_points={
                'console_scripts': ['boris=boris.cli:main']
            }
        )