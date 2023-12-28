from setuptools import setup

setup(
    name='pandas_splitter',
    version='0.0.1',
    description='Pandas file splitter by chunks without datetime intersection',
    url='https://github.com/TigProg/pandas_splitter',
    python_requires=">=3.11",
    install_requires=['pandas == 2.1.4', 'pytest == 7.4.3', 'numpy == 1.26.2'],
)
