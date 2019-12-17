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
      install_requires=['numpy==1.15.4',
                        'tensorflow==1.15.0',
                        'networkx==2.2',
                        'scipy==1.1.0'
                        ],
      package_data={'gcn': ['README.md']},
      packages=find_packages())