from setuptools import setup, find_packages
from os.path import dirname, realpath


def _read_requirements_file():
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


setup(
    name='Pretend-Benign',
    version='0.1.0',

    packages=find_packages(),
        
    install_requires=_read_requirements_file(),
    
    author='Hongwei Lin',
    description='Pretend Benign: A Stealthy Adversarial Attack by Exploiting Vulnerabilities in Cooperative Perception',
    python_requires='>=3.8',
)