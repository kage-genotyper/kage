[![PyPI version](https://badge.fury.io/py/kage-genotyper.svg)](https://badge.fury.io/py/kage-genotyper)
![example workflow](https://github.com/ivargr/kage/actions/workflows/install-and-test.yml/badge.svg)
[![DOI](https://zenodo.org/badge/251419423.svg)](https://zenodo.org/badge/latestdoi/251419423)


## KAGE: *K*mer-based *A*lignment-free *G*raph G*e*notyper
KAGE is a tool for efficiently genotyping short SNPs and indels from short genomic reads.

A manuscript describing the method [can be found here](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02771-2).

## Installation
KAGE requires Python 3, and can be installed using Pip: 
```
pip install kage-genotyper
```

Test that the installation worked:

```bash
kage test 
```

The above will perform genotyping on some dummy data and should finish without any errors. 


## How to run
**KAGE** is easy and fast to use once you have indexes built for the variants you want to genotype. However, building these indexes can take some time. Therefore, we have prebuilt indexes for 1000 Genomes Projects variants (allele frequency > 0.1%), which can be [downloaded from here](https://zenodo.org/record/6674055/files/index_2548all.npz?download=1).

If you want to make your own indexes for your own reference genome and variants, you should use the KAGE Snakemake pipeline which can [be found here](https://github.com/ivargr/genotyping-benchmarking). Feel free to contact us if you want help making these indexes for your desired variants.

Once you have an index of the variants you want to genotype, running KAGE is straight-forward:

### Step 1: Map fasta kmers to the pangenome index:
```python
kmer_mapper map -b index -f reads.fa -o kmer_counts
```

In the above example, the index specified by `-b` is an index bundle (see explanation above).

The kmer mapper works with .fa and .fq files. It can also takes gzipped-files, but for now this is a bit experimentally and may be a bit slow (it is using BioNumPy's parser which is under development).


### Step 2: Do the genotyping
Count kmers:
```bash
kage genotype -i index -c kmer_counts --average-coverage 15 -o genotypes.vcf
```

Make sure to set `--average-coverage` to the expected average coverage of your input reads. The resulting predicted genotypes will be written to the file specified by `-o`.


Note:
KAGE puts data and arrays in shared memory to speed up computation. It automatically frees this memory when finished, but KAGE gets accidentally killed or stops before finishing, you might end up with allocated memory not being freed. You can free this memory by calling `kage free_memory`.


## Recent changes and future plans
Recent changes:
* June 2022: Release of version for manuscript in Genome Biology
* October 7 2022: Minor update. Now using [BioNumPy](https://gitub.com/uio-bmi/bionumpy) do parse input files and hash kmers.

Future plans:
* Ultra-fast genotyping using GPU (a master student at Sandve lab is currently working on this)
* Support for structural variation


## Support
Please post an issue or email ivargry@ifi.uio.no if you encounter any problems or have any questions.