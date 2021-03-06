import setuptools

from mkb.__version__ import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setuptools.setup(
    name="mkb",
    version=f"{__version__}",
    author="Raphael Sourty",
    author_email="raphael.sourty@gmail.com",
    description="mkb",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raphaelsty/mkb",
    packages=setuptools.find_packages(),
    install_requires=required,
    package_data={
        "mkb": [
            "datasets/countries_s1/*.csv",
            "datasets/countries_s1/*.json",
            "datasets/countries_s2/*.csv",
            "datasets/countries_s2/*.json",
            "datasets/countries_s3/*.csv",
            "datasets/countries_s3/*.json",
            "datasets/fb13/*.csv",
            "datasets/fb13/*.json",
            "datasets/fb15k/*.csv",
            "datasets/fb15k/*.json",
            "datasets/fb15k237/*.csv",
            "datasets/fb15k237/*.json",
            "datasets/kinship/*.csv",
            "datasets/kinship/*.json",
            "datasets/nations/*.csv",
            "datasets/nations/*.json",
            "datasets/nell995/*.csv",
            "datasets/nell995/*.json",
            "datasets/semanlink/*.csv",
            "datasets/semanlink/*.json",
            "datasets/umls/*.csv",
            "datasets/umls/*.json",
            "datasets/wn11/*.csv",
            "datasets/wn11/*.json",
            "datasets/wn18/*.csv",
            "datasets/wn18/*.json",
            "datasets/wn18rr/*.csv",
            "datasets/wn18rr_text/*.csv",
            "datasets/wn18rr/*.json",
            "datasets/yago310/*.csv",
            "datasets/yago310/*.json",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
