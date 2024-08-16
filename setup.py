from setuptools import find_namespace_packages, find_packages, setup

from codecs import open
from os import path

HERE = path.abspath(path.dirname(__file__))
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyfus',
    version='0.1.4',
    description='Open source framework for functional ultrasound imaging data analysis',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://foss4fus.readthedocs.io",
    author='Théo Lambert @ Neuro-Electronics Research Flanders',
    author_email="theo.lambert@nerf.be",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    include_package_data=True,
    test_suite='tests',
    install_requires=[
        "numpy==1.23.5",
        "scipy==1.10.0",
        "pandas==1.5.2",
        "mat73==0.63",
        "pynrrd==1.0.0",
        "scikit-image==0.19.3",
        "scikit-learn==1.2.2",
        "seaborn==0.12.2",
        "Pillow==9.3.0",
        "nibabel==5.1.0",
        "openpyxl==3.1.4"
    ],
    packages=find_packages(include=['pyfus']),
    #packages=find_namespace_packages(where='foss4fus'),
    #package_dir={"": "foss4fus"},
    package_data={
        "pyfus": ["pyfus/atlases/atlases_lists/regions_ccf_v3_100_nolayersnoparts.txt", "pyfus/atlases/atlases_npy/atlas_ccf_v3_100_nolayersnoparts.npy", "pyfus/atlases/atlases_npy/atlas_ccf_v3_100_contours.npy"],
    }
)
