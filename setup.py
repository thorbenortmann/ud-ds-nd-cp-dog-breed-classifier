from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='dog_breed_classifier',
    packages=['dog_breed_classifier'],
    include_package_data=True,
    install_requires=requirements
)
