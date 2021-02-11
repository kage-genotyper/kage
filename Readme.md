

## Installation
```
git clone git@github.com:ivargr/obgraph.git
cd obgraph
python3 -m pip install -e .
```

Test that the installation worked:

```bash
alignment_free_graph_genotyper test -g Genotyper
```

The above should return genotyping accuracy 1.0.


Test on some real builtin data (from chr 1):
```bash
alignment_free_graph_genotyper test -g Genotyper -T real
```