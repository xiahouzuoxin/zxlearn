import re
from setuptools import setup, find_packages

# Read the long description from the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def find_version():
    with open("torchrec/__init__.py", "r", encoding="utf-8") as rf:
        version_file = rf.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(
    name='torchrec',
    version=find_version(),
    author='zuoxin.xiahou',
    author_email='xiahouzuoxin@163.com',
    description='A small pytorch implementation for ctr prediction in recommendation system for small companies',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xiahouzuoxin/torchrec",
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        'numpy',
        'pandas',
        'sklearn',
        'torch',
        'uvicorn',
        'fastapi',
        'pydantic'
    ],
    python_requires='>=3.10',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)