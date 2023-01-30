[![PyPI version](https://badge.fury.io/py/kage-genotyper.svg)](https://badge.fury.io/py/kage-genotyper)
![example workflow](https://github.com/ivargr/kage/actions/workflows/install-and-test.yml/badge.svg)
[![DOI](https://zenodo.org/badge/251419423.svg)](https://zenodo.org/badge/latestdoi/251419423)


## KAGE: *K*mer-based *A*lignment-free *G*raph G*e*notyper
KAGE is a tool for efficiently genotyping short SNPs and indels from short genomic reads.

As of version 0.1.11, KAGE also supports **GPU-acceleration** (referred to as **GKAGE**) and is able to genotype a human sample in a minutes with a GPU having as little as 4 GB of RAM. See guide for running with GPU-acceleration further down.

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


## Using KAGE with GPU-support (GKAGE)

As of version 0.1.11, KAGE supports GPU-acceleration for GPUs supporting the CUDA-interface. You will need to have CUDA installed on your system along with CuPy (not automatically installed as part of KAGE). Follow these steps to run KAGE with GPU-support:

1) Make sure you have [CUDA installed](https://developer.nvidia.com/cuda-downloads) and working.
2) Install a [CuPy](https://docs.cupy.dev/en/stable/install.html) version that matches your CUDA installation.
3) Install [Cucounter](https://github.com/jorgenwh/cucounter)
4) Run kmer_mapper with `--gpu True` to tell kmer_mapper to use the GPU. If you want to save memory, you can also use a kmer index that does not have reverse complements. Kmer mapper will then compute the reverse complements from your reads. To this by specifying `-i kmer_index_only_variants_without_revcomp.npz`. For humans you can [use this index](https://zenodo.org/record/7582195/files/kmer_index_only_variants_without_revcomp.npz?download=1).
5) Run `kage genotype` as normal (kage genotype is already very fast in CPU-mode and does not use GPU-acceleration now).

Note: GKAGE has been tested to work with GPUs with 4 GBs of RAM.


## Recent changes and future plans
Recent changes:
* Janyary 30 2023: Release of GPU support (version 0.0.30).
* October 7 2022: Minor update. Now using [BioNumPy](https://gitub.com/uio-bmi/bionumpy) do parse input files and hash kmers.
* June 2022: Release of version for manuscript in Genome Biology


## Support
Please post an issue or email ivargry@ifi.uio.no if you encounter any problems or have any questions.
