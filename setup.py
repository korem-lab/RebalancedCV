import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join( os.path.dirname(__file__), fname) ).read()


setup(
    name='RebalancedCV',
    version='0.0.1',
    author='George Austin', 
    author_email='gia2105@columbia.edu', 
    url='https://github.com/korem-lab/RebalancedCV',
    packages=['rebalancedcv'],
    include_package_data=True,
    install_requires=['numpy', 
                      'scikit-learn' 
                     ]
)
