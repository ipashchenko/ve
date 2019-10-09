try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup


setup(
    name='vlbi_errors',
    version='0.1',
    author='Ilya Pashchenko',
    author_email='in4pashchenko@gmail.com',
    packages=['vlbi_errors', 'tests'],
    scripts=[],
    url='https://github.com/ipashchenko/vlbi_errors',
    license='MIT',
    description='Tools for acscesing uncertaintes of VLBI results',
    long_description=open('README.rst').read(),
    install_requires=[
        "numpy >= 1.7.2",
        "astropy", 'scipy', 'scikit-image', 'scikit-learn', 'BeautifulSoup'
    ],)
