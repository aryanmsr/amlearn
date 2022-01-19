from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Education',
  'Operating System :: MacOS :: MacOS X',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3.8'
]
 
setup(
  name='amlearn-v0.0.1',
  version='0.0.1',
  description='A very basic machine learning library.',
  long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
  url='',  
  author='Aryan Mishra',
  author_email='aryanmsr@gmail.com',
  license='MIT', 
  classifiers=classifiers,
  keywords='machine_learning_lib', 
  packages=find_packages(include=['Linear_Regression', 'Linear_Regression.*', 'Logistic_Regression', 'Logistic_Regression.*']),
  install_requires=[''],
)