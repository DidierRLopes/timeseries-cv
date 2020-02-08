from setuptools import setup

setup(
    # Needed to silence warnings
    name='TimeSeriesCrossValidation',
    url='',
    author='Didier Lopes',
    author_email='dro.lopes@campus.fct.unl.pt',
    # Needed to actually package something
    packages=['TimeSeriesCrossValidation.splitTrain'],
    # Needed for dependencies
    install_requires=['numpy'],
    # *strongly* suggested for sharing
    version='0.1',
    license='MIT',
    description='An example of a python package from pre-existing code',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.rst').read(),
    # if there are any scripts
    scripts=['scripts/testSplit.py'],
)
