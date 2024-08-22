import logging

import numpy as np
from kage.io import VcfWithInfoBuffer

logging.basicConfig(level=logging.INFO)
import bionumpy as bnp
from kage.util import log_memory_usage_now

f = bnp.open("test2.vcf", lazy=True)
log_memory_usage_now("Start")
all = []
for chunk in f.read_chunks():
    print(chunk.genotypes)
    all.append(chunk)

all = np.concatenate(all)

log_memory_usage_now("End")
print(all.genotypes)





