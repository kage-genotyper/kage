from kage.variants import VcfVariant, VcfVariants

def test_position_index():
    variants = VcfVariants(
        [
            VcfVariant(1, 4),
            VcfVariant(1, 7),
            VcfVariant(1, 8),
            VcfVariant(1, 8),
            VcfVariant(3, 8),
            VcfVariant(3, 9),
            VcfVariant(3, 11),
        ]
    )

    variants.make_position_index()

    v = variants.get_variants_in_region(1, 4, 8)
    print(v)


test_position_index()