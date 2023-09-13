import pytest
from kage.io import VcfWriter
from kage.preprocessing.variants import Variants

@pytest.fixture
def variants():
   variants = Variants.from_entry_tuples(
       [
           ("chr1", "")
       ]
   )

def test_vcf_writer():
