from setuptools import setup, find_packages

"""
python3 -m unittest
vim setup.py
rm -rf dist/
python3 setup.py sdist bdist_wheel
twine upload --repository pypi dist/*
"""


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name="papertown",
    version="0.1", #.10.14",
    license="MIT",
    author="Kimio Kuramitsu",
    description="The PaperTown LLM Project",
    url="https://github.com/kuramitsulab/papertown",
    packages=["papertown"],
    package_dir={"papertown": "papertown"},
    package_data={"papertown": ["*/*"]},
    install_requires=_requires_from_file("requirements.txt"),
    entry_points={
        "console_scripts": [
            "papertown_store=papertown.cli:main_store",
	    "papertown_update=papertown.cli:main_update",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Education",
    ],
)
