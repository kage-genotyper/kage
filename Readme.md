[![PyPI version](https://badge.fury.io/py/kage-genotyper.svg)](https://badge.fury.io/py/kage-genotyper)
![example workflow](https://github.com/ivargr/kage/actions/workflows/install-and-test.yml/badge.svg)
[![DOI](https://zenodo.org/badge/251419423.svg)](https://zenodo.org/badge/latestdoi/251419423)


## Update November 20 2023
* KAGE now has experimental support for genotyping structural variants. 
* The indexing process has been rewritten and indexing is now much faster and requires less memory. There is now one single command for creating the indexes from an input vcf.
* Various minor fixes for improved genotyping accuracy on SNPs and short indels.


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

Variants can be biallelic or multiallelic and both SNPs/indels and structural variants are supported. Note however that all variants must have actual sequences in the ref and alt fields. Genotypes should be phased (e.g. `0|0`, `0|1` or `1|1`) and there should ideally be few missing genotypes (e.g. `.|.` or `.`). If there are structural variants present, KAGE will prioritize those, meaning that accuracy on SNPs and indels may be lower (especially for SNPs and indels that are covered by SVs). If your aim is to only genotype SNPs and indels, you should not not include SVs in your vcf.

### Step 1: Build an index of the variants you want to genotype
Building an index is somewhat time consuming, but only needs to be done once for each set of variants you want to genotype. Indexing time scales approximately linearly with number of variants and the size of the reference genome. Creating an index for a human pangenome with 30 million variants and 2000-3000 individuals should finish in less than a day. It's always a good idea to start out with a smaller set of variants, e.g. a single chromosome first to see if things work as expected.

```bash
kage index -r reference.fa -v variants.vcf.gz -o index -k 31
```

The above command will create an `index.npz` file.

### Step 2: Genotype
Genotyping with kage is extremely fast once you have an index:

```bash
kage genotype -i index -f reads.fq.gz -t 16 --average-coverage 30 -k 31
```

Note:
* `-k` must be set to the same that was used when creating the index
* `--average-coverage` should be set to the expected average coverage of your input reads (doesn't need to be exact)
* KAGE puts data and arrays in shared memory to speed up computation. It automatically frees this memory when finished, but KAGE gets accidentally killed or stops before finishing, you might end up with allocated memory not being freed. You can free this memory by calling `kage free_memory`.

## KAGE works even better with GLIMPSE
KAGE uses information from the population to improve accuracy, a bit similarily to imputation. However, the model used by KAGE is quite simple. It works well for SNPs and indels, but for SVs, we have found that using GLIMPSE for the imputation-step works much better. To run KAGE with GLIMPSE instead of the builtin KAGE imputation, simpy add `--glimpse variants.vcf.gz` when running `kage genotype`. KAGE will automatically install GLIMPSE by downloading binaries and run GLIMPSE for you. One should expect some longer runtime, but not much.

Note: GLIMPSE requires that you have BCFTools installed.

### Prebuilt indexes

You can find some prebuilt indexes here (coming soon):

* Draft Human Pangenome (47 individuals, SVs and SNPs/indels)
* 1000 genomes SNPs/indels, 2548 individuals


## Using KAGE with GPU-support (GKAGE)

As of version 0.1.11, KAGE supports GPU-acceleration for GPUs supporting the CUDA-interface. You will need to have CUDA installed on your system along with CuPy (not automatically installed as part of KAGE). Follow these steps to run KAGE with GPU-support:

1) Make sure you have [CUDA installed](https://developer.nvidia.com/cuda-downloads) and working.
2) Install a [CuPy](https://docs.cupy.dev/en/stable/install.html) version that matches your CUDA installation.
3) Install [Cucounter](https://github.com/jorgenwh/cucounter)
4) Run kage with `--gpu True` to tell KAGE to use the GPU. 

Note: GKAGE has been tested to work with GPUs with 4 GBs of RAM.


## Recent changes and future plans
Recent changes:
* November 20 2023: Indexing process rewritten and experimental support for structural variation.
* January 30 2023: Release of GPU support (version 0.0.30).
* October 7 2022: Minor update. Now using [BioNumPy](https://gitub.com/uio-bmi/bionumpy) do parse input files and hash kmers.
* June 2022: Release of version for manuscript in Genome Biology


## Support
Please post an issue or email ivargry@ifi.uio.no if you encounter any problems or have any questions.
