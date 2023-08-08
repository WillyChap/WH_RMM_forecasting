from setuptools import setup, find_packages

setup(
    name='WH_RMM_Forecasting',
    version='0.1.0',
    author='William E. Chapman',
    author_email='wchapman@ucar.edu',
    description='A package to create the forecasting of MJO indices',
    license='GNU General Public License v3',
    packages=find_packages(include='WH_RMM_forecasting*'),
    install_requires=[
        'pyyaml',
        'pandas',#pandas 2.0? 
        'numpy',
        'matplotlib',
        'xarray',
        'eofs',
        'datetime',
        'scipy',
        'netCDF4'
    ],
    extras_require={
        'full_func': ['jupyter','dask'],
        'dev': ['build', 'pytest', 'pytest-pep8'],
      },
    package_data={'WH_RMM_forecasting': ['*/*.nc']},
)