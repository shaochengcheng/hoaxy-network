from os.path import dirname, join
from setuptools import setup, find_packages

with open(join(dirname(__file__), 'hnetwork/VERSION'), 'rb') as f:
    version = f.read().decode('ascii').strip()

setup(
    name='hoaxy-network',
    version=version,
    url='http://cnets.indiana.edu',
    description='Network analysis of hoaxy data',
    long_description=open('README.md').read(),
    author='Chengcheng Shao',
    maintainer='Chengcheng Shao',
    maintainer_email='shaoc@indiana.edu',
    license='GPLv3',
    entry_points={'console_scripts': ['hnetwork = hnetwork.cmdline:main']},
    packages=find_packages(exclude=('tests', 'tests.*')),
    include_package_data=True,
    zip_safe=True,
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: GPL :: Version 3',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    install_requires=[
        'networkx',
        'pandas',
        'docopt>=0.6.2',
        'schema',
    ],)
