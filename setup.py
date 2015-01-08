# setup.py build file

from __future__ import print_function

DESCRIPTION = "Berkeley Vision Stats"
LONG_DESCRIPTION = """\
Berkeley Vision Stats is a project to analyze the statistics of the visual world during natural
viewing.
"""

NAME = "BerkeleyVisionStats"
AUTHOR = "Bill Sprague and Emily Cooper"
AUTHOR_EMAIL = "bill.sprague@berkeley.edu"
DOWNLOAD_URL = "https://github.com/eacooper/BerkeleyVisionStats.git"
LICENSE = "MIT"
VERSION = "0.1.dev0"

from setuptools import setup

if __name__ == '__main__':
    setup(name=NAME,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        download_url=DOWNLOAD_URL,
        version=VERSION,
        packages=['bvs', 'bvs.utils', 'bvs.cli', 'bvs.eyelink-gazeconversion',
                    'bvs.eyelinkfileparsing'],
        classifiers=[
                     'Intended Audience :: Science/Research',
                     'Programming Language :: Python :: 2.7',
                     #'Programming Language :: Python :: 3.3',
                     #'Programming Language :: Python :: 3.4',
                     'License :: OSI Approved :: MIT License',
                     'Topic :: Scientific/Engineering',
                     'Operating System :: POSIX',
                     'Operating System :: Unix',
                     'Operating System :: MacOS'],
        install_requires=['numpy', 'pandas', 'Click'],
        # scripts=['bin/bvs'],
        entry_points={
                'console_scripts': ['bvs=bvs.cli:main']
            }
        )