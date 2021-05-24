from setuptools import setup

setup(
    name='timeseries-cv',
    url='https://github.com/DidierRLopes',
    author='Didier Rodrigues Lopes',
    author_email='dro.lopes@campus.fct.unl.pt',
    packages=['tsxv'],
    install_requires=['numpy'],
    version='0.1.4',
    license='LICENSE.txt',
    description='Implementation of Cross-Validation techniques (Forward Chaining, K-Fold, Group K-Fold) to Time-Series',
    long_description=open('README.md').read(),
)
