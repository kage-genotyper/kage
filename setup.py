from setuptools import setup, Extension
#from distutils.extension import Extension
#from Cython.Distutils import build_ext

ext_modules=[
      Extension("alignment_free_graph_genotyper.cython_chain_genotyper",
                ["alignment_free_graph_genotyper/cython_chain_genotyper.pyx"],
                libraries=["m"],
                extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
                extra_link_args=['-fopenmp'],
                )]

setup(name='alignment_free_graph_genotyper',
      version='0.0.1',
      description='Alignment-free Graph Genotyper',
      url='http://github.com/ivargr/alignment_free_graph_genotyper',
      author='Ivar Grytten',
      author_email='',
      license='MIT',
      packages=["alignment_free_graph_genotyper"],
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
            'console_scripts': ['alignment_free_graph_genotyper=alignment_free_graph_genotyper.command_line_interface:main']
      },
      #cmdclass = {"build_ext": build_ext},
      ext_modules = ext_modules
)
