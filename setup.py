from setuptools import setup, Extension, find_packages

setup(name='kage-genotyper',
      version='2.0.3',
      description='KAGE',
      long_description_content_type="text/markdown",
      url='http://github.com/ivargr/kage',
      author='Ivar Grytten',
      author_email='',
      license='MIT',
      packages=find_packages(include=['kage', 'kage.*']),
      zip_safe=False,
      install_requires=['numpy', 'tqdm', 'pyfaidx', 'pathos', 'cython', 'scipy',
                        'obgraph>=0.0.35',
                        'graph_kmer_index>=0.0.29',
                        'kmer_mapper>=0.0.32',
                        'graph_read_simulator>=0.0.7',
                        'shared_memory_wrapper>=0.0.28',
                        'bionumpy>=1.0.0',
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
