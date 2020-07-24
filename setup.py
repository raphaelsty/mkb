import setuptools

from kdmkb.__version__ import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kdmkb",
    version=f"{__version__}",
    author="Raphael Sourty",
    author_email="raphael.sourty@gmail.com",
    description="kdmkb",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raphaelsty/kdmkb",
    packages=setuptools.find_packages(),
    package_data={
        'kdmkb': [
            'datasets/wn18rr/*.csv',
            'datasets/wn18rr/*.json',
            'datasets/fb15k237/*.csv',
            'datasets/fb15k237/*.json',
            'datasets/yago310/*.csv',
            'datasets/yago310/*.json'
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],

    python_requires='>=3.6',
)
