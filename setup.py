from setuptools import setup, Extension

setup(name='kage-genotyper',
      version='0.1.6',
      description='KAGE',
      long_description_content_type="text/markdown",
      url='http://github.com/ivargr/kage',
      author='Ivar Grytten',
      author_email='',
      license='MIT',
      packages=["kage"],
      zip_safe=False,
      install_requires=['numpy', 'tqdm', 'pyfaidx', 'pathos', 'cython', 'scipy',
                        'obgraph>=0.0.30',
                        'graph_kmer_index>=0.0.22',
                        'kmer_mapper>=0.0.22',
                        'graph_read_simulator>=0.0.7',
                        'shared_memory_wrapper>=0.0.18',
                        'bionumpy>=0.2.7',
                        'npstructures>=0.2.2'
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