[![PyPI version](https://badge.fury.io/py/kage-genotyper.svg)](https://badge.fury.io/py/kage-genotyper)
![example workflow](https://github.com/ivargr/kage/actions/workflows/install-and-test.yml/badge.svg)
[![DOI](https://zenodo.org/badge/251419423.svg)](https://zenodo.org/badge/latestdoi/251419423)


## Update October 16 2023
* KAGE can now genotype structural variants
* The indexing process has been rewritten and indexing is now much faster and requires less memory


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
You will need:
* A reference genome in fasta format
* A set of variants with genotypes of known individuals in vcf-format (`.vcf` or `.vcf.gz`)

Variants can be biallelic or multiallelic and both SNPs/indels and structural variants are supported. Note however that all variants must have actual sequences in the ref and alt fields. Genotypes should be phased and there should ideally be few missing genotypes.

### Step 1: Build an index of the variants you want to genotype
Building an index is time consuming, but only needs to be done once for each set of variants you want to genotype. Indexing time scales approximately linearly with number of variants and the size of the reference genome. Creating an index for a human pangenome with 30 million variants should take approximately a day or so.

```bash
kage index -r reference.fa -v variants.vcf.gz -o index -k 31
```

### Step 2: Genotype
Genotyping with kage is extremely fast once you have an index:

```bash
kage genotype -i index -f reads.fq.gz -t 16 --average-coverage 30 -k 31
```

Note:
* `-k` must be set to the same used when creating the index
* `--average-coverage` should be set to the expected average coverage of your input reads (doesn't need to be exact)
* KAGE puts data and arrays in shared memory to speed up computation. It automatically frees this memory when finished, but KAGE gets accidentally killed or stops before finishing, you might end up with allocated memory not being freed. You can free this memory by calling `kage free_memory`.

### Prebuilt indexes

You can find some prebuilt indexes here (coming soon):

* 1000 genomes SNPs/indels, 2548 individuals
* 1000 genomes SVs
* 1000 genomes SNPs/indels + SVs


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
* October 16 2023: Indexing process rewritten and support for structural variation.
* January 30 2023: Release of GPU support (version 0.0.30).
* October 7 2022: Minor update. Now using [BioNumPy](https://gitub.com/uio-bmi/bionumpy) do parse input files and hash kmers.
* June 2022: Release of version for manuscript in Genome Biology


## Support
Please post an issue or email ivargry@ifi.uio.no if you encounter any problems or have any questions.
