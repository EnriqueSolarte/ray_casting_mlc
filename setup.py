from setuptools import find_packages, setup

with open("./requirements.txt", "r") as f:
    requirements = [l.strip() for l in f.readlines() if len(l.strip()) > 0]

setup(
    name="ray-casting-mlc",
    version="1.0",
    packages=find_packages(),
    install_requires=requirements,
    package_data={
                  "ray-casting-mlc": ["models/**", "config/**"]
                },
    author="Enrique Solarte",
    author_email="enrique.solarte.pardo@gmail.com",
    description=("Ray casting algorithm mlc - ECCV2024"),
    license="BSD",
)
