import setuptools
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'robo_gym'))
from version import VERSION

setuptools.setup(name='robo-gym',
      version=VERSION,
      description='robo-gym: an open source toolkit for Distributed Deep Reinforcement Learning on real and simulated robots.',
      url='https://github.com/jr-robotics/robo-gym',
      author="Matteo Lucchi, Friedemann Zindler, Bernhard Reiterer, Thomas Gallien, Benjamim Breiling",
      author_email="bernhard.reiterer@joanneum.at",
      packages=setuptools.find_packages(),
      include_package_data=True,
      classifiers=[
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
      ],
      install_requires=[
      'gymnasium',
      'robo-gym-server-modules',
      'numpy',
      'scipy',
      'pyyaml'
      ],
      python_requires='>=3.8',
      scripts = ['bin/run-rs-side-standard']
)
