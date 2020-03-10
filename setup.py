from setuptools import setup

setup(
    name='TimeSeriesCrossValidation',
    url='https://github.com/DidierRLopes',
    author='Didier Rodrigues Lopes',
    author_email='dro.lopes@campus.fct.unl.pt',
    packages=['TimeSeriesCrossValidation'],
    install_requires=['numpy'],
    version='0.3',
    license='LICENSE.txt',
    description='Implementation of Cross-Validation techniques (Forward Chaining, K-Fold, Group K-Fold) to Time-Series',
    long_description=open('README.md').read(),
)
