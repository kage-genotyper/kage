from kage.analysis.genotype_accuracy import GenotypeAccuracy, read_vcf_with_genotypes, normalize_genotype, IndexedGenotypes


def test_read_vcf():
    data = read_vcf_with_genotypes("../example_data/vcf_with_genotypes.vcf")
    print(data)
    print(data.genotype)


def test_normalize_genotype():
    assert normalize_genotype("0/0") == "0/0"
    assert normalize_genotype("5|1") == "1/5"
    assert normalize_genotype(".|1") == "./1"
    assert normalize_genotype(".") == "0/0"
    assert normalize_genotype("0/1:0.5") == "0/1"


def test_genotype_accuracy():
    truth = IndexedGenotypes({
        "1": "0/0",
        "2": "0/1",
        "10": "1|0",
    })

    sample = IndexedGenotypes({
        "1": "1/0",
        "2": "0/1",
        "10": "0/0"
    })

    accuracy = GenotypeAccuracy(truth, sample)
    assert accuracy.false_negative == 1
    assert accuracy.true_positive == 1
    assert accuracy.true_negative == 0
    assert accuracy.false_negative == 1

    assert accuracy.recall() == 0.5
    assert accuracy.precision() == 0.5

