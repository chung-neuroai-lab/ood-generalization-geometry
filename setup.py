from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ood-generalization-geometry",
    version="0.1.0",
    author="Chung Neuroai Lab",
    description="Code for 'Diagnosing Generalization Failures from Representational Geometry'",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chung-neuroai-lab/ood-generalization-geometry",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
