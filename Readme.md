
## kage: *K*mer-based *A*lignment-free *G*raph G*e*notyper
KAGE is a tool for efficiently genotyping short SNPs and indels from short genomic reads.


## Installation
```
pip install kage-genotyper
```
.. or
```
conda install kage
```

Test that the installation worked:

```bash
kage test 
```

The above will perform genotyping on some dummy data and should finish without any errors. 


## How to run
**kage** is easy and fast to use once you have indexes built for the variants you want to genotype. However, building these indexes can take some time. Therefore, we have prebuilt indexes for 100 Genomes Projects variants, which can be [downloaded from here](..).

If you want to make your own indexes for your own reference genome and variants, you should use the kage Snakemake pipeline which can [be found here](..).

Once you have an index of the variants you want to genotype, running kage is straight-forwards:

Count kmers:
```bash
kage count -i 1000genomes.npz ...

```