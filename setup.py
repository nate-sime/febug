import sys

from setuptools import setup

if sys.version_info < (3, 10):
    print("Python 3.10 or higher required, please upgrade.")
    sys.exit(1)

VERSION = "0.9.0"

REQUIREMENTS = ["pyvista", "fenics-dolfinx>=0.9.0.dev0"]

setup(
    name='febug',
    version=VERSION,
    packages=['febug',
              'febug.dolfinx',
              'febug.dolfinx.fem',
              'febug.dolfinx.nls'],
    # ext_modules=[CMakeExtension('febug.cpp')],
    # cmdclass=dict(build_ext=CMakeBuild),
    install_requires=REQUIREMENTS,
    url='',
    license='MIT',
    author='Nate Sime',
    author_email='nsime@carnegiescience.edu',
    description='Utility functions for debugging dolfinx gotchas'
)
