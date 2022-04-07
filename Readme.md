[![PyPI version](https://badge.fury.io/py/kage-genotyper.svg)](https://badge.fury.io/py/kage-genotyper)
![example workflow](https://github.com/ivargr/kage/actions/workflows/install-and-test.yml/badge.svg)

## KAGE: *K*mer-based *A*lignment-free *G*raph G*e*notyper
KAGE is a tool for efficiently genotyping short SNPs and indels from short genomic reads.


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
**KAGE** is easy and fast to use once you have indexes built for the variants you want to genotype. However, building these indexes can take some time. Therefore, we have prebuilt indexes for 1000 Genomes Projects variants (allele frequency > 0.1%), which can be [downloaded from here](https://zenodo.org/record/5786313/files/index_2548all.npz).

If you want to make your own indexes for your own reference genome and variants, you should use the KAGE Snakemake pipeline which can [be found here](https://github.com/ivargr/genotyping-benchmarking). Feel free to contact us if you want help making these indexes for your desired variants.

Once you have an index of the variants you want to genotype, running KAGE is straight-forward:

### Step 1: Map fasta kmers to the pangenome index:
```python
kmer_mapper map -b index -f reads.fa -o kmer_counts -l 150
```

The Kmer Mapper requires for now a two-line fasta file. If you have a fastq file, you should convert that to fasta before mapping (e.g. by using [Seqtk](https://github.com/lh3/seqtk)).

Note: Make sure l is the max read length of your input reads, and not any lower than that. The index specified by `-b` is an index bundle (see explanation above).


### Step 2: Do the genotyping
Count kmers:
```bash
kage genotype -i index -c kmer_counts --average-coverage 15 -o genotypes.vcf
```

Make sure to set `--average-coverage` to the expected average coverage of your input reads. The resulting predicted genotypes will be written to the file specified by `-o`.


Note:
KAGE puts data and arrays in shared memory to speed up computation. It automatically frees this memory when finished, but KAGE gets accidentally killed or stops before finishing, you might end up with allocated memory not being freed. You can free this memory by calling `kage free_memory`.



## Running KAGE in "Vanilla mode"
KAGE uses information from a population and a model of expected kmer counts in order to improve genotyping accuracy. It is possible to skip all of this, and just run KAGE in "vanilla mode". This can be useful if you want to do your own imputation from the genotype likelyhoods that KAGE outputs, or if you want a simplified genotyping with where no prior information is being used.

KAGE can be run on both mapped reads (only [vg](https://github.com/vgteam/vg) is supported for now) or kmer counts (only [kmer_mapper](https://github.com/ivargr/kmer_mapper) is supported). When using kmers, a model of expected kmer counts should be used in order to get decent accuracy. With mapped reads, it is possible to genotype without any model, and this is a very simple option that doesn't require much work prior to genotyping. The options are detailed below.

### Option 1: Using mapped reads with vg
We assume you already have a `graph.xg` that represents all the variants you want to genotype and a `variants.vcf.gz` that the graph has been built from.

#### Step 1: Make an obgraph

We start by converting this graph to GFA, and creating a [obgraph](https://github.com/ivargr/obgraph):

```bash
vg view -g graph.xg > graph.gfa
obgraph from_gfa -g graph.gfa -o graph_tmp.npz
```

#### Step 2: Add indel nodes
KAGE needs variants in the graph to always be represented with two nodes, one for the reference and one for the variant allele. We do this by adding empty dummy nodes for deltions and insertions:

```bash
obgraph add_indel_nodes -g graph_tmp.npz -v variants.vcf.gz -o graph.npz 
```

We also need a "mapping" from variants to nodes in the graph:
```python
obgraph make_variant_to_nodes -g graph.npz -v variants.vcf.gz -o variant_to_nodes.npz 
```

#### Step 3: Map with vg giraffe and get node counts
We map with vg giraffe and specify gaf as the output format:
```bash
vg giraffe .... -o gaf > mapped_reads.gaf 
```

We want to count how many of the reads support nodes in the graph. Note that we here need an "edge-mapping", which was created in step 2 when we created indel nodes. Change --min-mapq and --min-score to what you want:

```bash
kage node_counts_from_gaf -g mapped_reads.gaf --min-mapq 30 --min-score 100 -o node_conts.npy -m obgraph.npz.edge_mapping
```

### Step 4: Genotype
We can now genotype using the node counts from step 3:

```python
kage genotype -c node_counts.npy -g variant_to_nodes.npz -v variants.vcf.gz --n-threads 8 --average-coverage 15 -o genotypes.vcf --sample-name-output SAMPLE
```

Change `--average-coverage` according to your experiment.





