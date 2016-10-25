from setuptools import setup

setup(name='esvalues',
      version='0.2',
      description='Estimates Expectation Shapley (ES) values for arbirary functions.',
      url='http://github.com/interpretable-ml/esvalues.py',
      author='Scott Lundberg',
      author_email='slund1@cs.washington.edu',
      license='MIT',
      packages=['esvalues'],
      install_requires=['numpy', 'scipy'],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
