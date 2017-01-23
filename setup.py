from setuptools import setup
from setuptools import find_packages

setup(name='gcn',
      version='1.0',
      description='Graph Convolutional Networks in Tensorflow',
      author='Thomas Kipf',
      author_email='thomas.kipf@gmail.com',
      url='https://tkipf.github.io',
      download_url='https://github.com/tkipf/gcn',
      license='MIT',
      install_requires=['numpy',
                        'tensorflow',
                        'networkx',
                        'scipy'
                        ],
      package_data={'gcn': ['README.md']},
      packages=find_packages())