from setuptools import setup, find_packages


setup(
    name='yaNEAT',
    version='0.0.1',
    description='Yet another NEAT implementation',
    author='Tim Clough',
    author_email='tmclough98@gmail.com',
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.0.0alpha0',
        'numpy'
    ]
)