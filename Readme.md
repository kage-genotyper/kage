
## kage: *K*mer-based *A*lignment-free *G*raph G*e*notyper


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


## How to run
**kage** is very easy and fast to use once you have indexes built for the variants you want to genotype. However, building these indexes can take some time. Therefore, we have prebuilt indexes for 100 Genomes Projects variants, which can be [downloaded from here](..).

If you want to make your own indexes for your own reference genome and variants, you should use the kage Snakemake pipeline which can [be found here](..).

Once you have an index of the variants you want to genotype, running kage is straight-forwards:

Count kmers:
```bash
kage count -i 1000genomes.index ...

```