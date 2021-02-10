from setuptools import setup

setup(name='alignment_free_graph_genotyper',
      version='0.0.1',
      description='Alignment-free Graph Genotyper',
      url='http://github.com/ivargr/alignment_free_graph_genotyper',
      author='Ivar Grytten',
      author_email='',
      license='MIT',
      packages=["alignment_free_graph_genotyper"],
      zip_safe=False,
      install_requires=['numpy', 'tqdm', 'pyfaidx'],
      include_dirs=["."],
      classifiers=[
            'Programming Language :: Python :: 3'
      ],
      entry_points={
            'console_scripts': ['alignment_free_graph_genotyper=alignment_free_graph_genotyper.command_line_interface:main']
      }
)
