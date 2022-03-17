from setuptools import setup, Extension

setup(name='kage-genotyper',
      version='0.0.19',
      description='KAGE',
      long_description_content_type="text/markdown",
      url='http://github.com/ivargr/kage',
      author='Ivar Grytten',
      author_email='',
      license='MIT',
      packages=["kage"],
      zip_safe=False,
      install_requires=['numpy', 'tqdm', 'pyfaidx', 'pathos', 'cython', 'scipy',
                        'obgraph>=0.0.9',
                        'graph_kmer_index>=0.0.15',
                        'kmer_mapper>=0.0.17',
                        'graph_read_simulator',
                        'shared_memory_wrapper>=0.0.7'
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