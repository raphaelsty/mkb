import setuptools

from kdmkr.__version__ import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kdmkr",
    version=f"{__version__}",
    author="Raphael Sourty",
    author_email="raphael.sourty@gmail.com",
    description="kdmkr",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raphaelsty/kdmkr",
    packages=setuptools.find_packages(),
    package_data={
        'kdmkr': [
            'datasets/wn18rr/*.csv',
            'datasets/wn18rr/*.json',
            'datasets/fb15k237/*.csv',
            'datasets/fb15k237/*.json'
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],

    python_requires='>=3.6',
)
