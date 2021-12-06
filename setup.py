from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()
    
# Setting up
setup(
    name="simpleconjoint",
    version='0.0.1',
    author="DATasso (Daniel Aguilera)",
    author_email="aguilera.t.daniel@gmail.com",
    description="A package to perform conjoint in Python",
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=["simpleconjoint"],
    license="MIT",      
    install_requires=['Cython>=0.22', 'numpy>=1.7', 'pandas>=1.2.4', 'pystan==2.19.1.1', 'XlsxWriter>=3.0.1'], 
    keywords=['python', 'conjoint', 'conjoint analysis', 'cbc', 'simple conjoint', 'simpleconjoint'],
    url="https://github.com/DATasso/simpleconjoint",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)