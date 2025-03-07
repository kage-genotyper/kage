from setuptools import setup, Extension, find_packages

setup(name='kage-genotyper',
      version='2.0.7',
      description='KAGE',
      long_description_content_type="text/markdown",
      url='http://github.com/ivargr/kage',
      author='Ivar Grytten',
      author_email='',
      license='MIT',
      packages=find_packages(include=['kage', 'kage.*']),
      zip_safe=False,
      install_requires=['numpy<2', 'tqdm', 'pyfaidx', 'pathos', 'cython', 'scipy',
                        'obgraph>=0.0.35',
                        'graph_kmer_index>=0.0.29',
                        'kmer_mapper>=0.0.38',
                        'npstructures>=0.2.16',
                        'graph_read_simulator>=0.0.7',
                        'shared_memory_wrapper>=0.0.32',
                        'bionumpy>=1.0.2',
                        'awkward',
                        'numba',
                        'ray',
                        'isal'
                        ],
      include_dirs=["."],
      classifiers=[
            'Programming Language :: Python :: 3'
      ],
      entry_points={
            'console_scripts': ['kage=kage.command_line_interface:main']
      }
)

""""
rm -rf dist
python3 setup.py sdist
twine upload --skip-existing dist/*

"""
