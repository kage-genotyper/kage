from setuptools import setup, Extension



with open("Readme.md", 'r') as f:
    long_description = f.read()

setup(name='kage',
      version='0.0.1',
      description='KAGE',
      long_description=long_description,
      url='http://github.com/ivargr/kage',
      author='Ivar Grytten',
      author_email='',
      license='MIT',
      packages=["kage"],
      zip_safe=False,
      install_requires=['numpy', 'tqdm', 'pyfaidx', 'pathos', 'cython', 'scipy',
                        'obgraph @ git+https://git@github.com/ivargr/obgraph@master#egg=obgraph',
                        'graph_kmer_index @ git+https://git@github.com/ivargr/graph_kmer_index@master#egg=graph_kmer_index'
                        ],
      include_dirs=["."],
      classifiers=[
            'Programming Language :: Python :: 3'
      ],
      entry_points={
            'console_scripts': ['kage=kage.command_line_interface:main']
      }

)
