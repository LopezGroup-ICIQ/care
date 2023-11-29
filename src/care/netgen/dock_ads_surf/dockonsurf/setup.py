import setuptools

with open("README.md", "r") as readme_fh:
    long_description = readme_fh.read()

install_requires = [
    "ase>=3.19.1",
    "hdbscan~=0.8.26",
    "matplotlib>=3.2.1",
    "networkx>=2.4",
    "numpy>=1.16.6",
    "pycp2k~=0.2.2",
    "pymatgen~=2020.11.11",
    "python-daemon~=2.2.4",
    "rdkit>=2019.9.3",
    "scikit-learn~=0.23.1",
]

setuptools.setup(
    name="dockonsurf",  # Replace with your own username
    version="0.0.1",
    author="Carles Martí",
    author_email="carles.marti2@gmail.com",
    description="Code to systematically find the most stable geometry for "
                "molecules on surfaces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/lch_interfaces/dockonsurf",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
