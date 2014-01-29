import os
import sys
import re

try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

# Handle encoding
major, minor1, minor2, release, serial = sys.version_info


def rd(filename):
    if major >= 3:
        f = open(filename, encoding="utf-8")
        r = f.read()
        f.close()
    else:
        f = open(filename)
        r = f.read()
        f.close()

        return r


vre = re.compile("__version__ = \"(.*?)\"")
m = rd(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "asc", "__init__.py"))
version = vre.findall(m)[0]


setup(
    name="asc",
    version=version,
    author="Ilya Pashchenko",
    author_email="in4pashchenko@gmail.com",
    scripts=[],
    packages=["asc"],
    license="MIT",
    description="Tools for automating post-correlation data processing at ASC",
    long_description=rd("README.rst") + "\n\n"
                    + "Changelog\n"
                    + "---------\n\n"
                    + rd("HISTORY.rst"),
    package_data={"": ["LICENSE", "AUTHORS.rst"]},
    include_package_data=True,
    install_requires=[],
)
