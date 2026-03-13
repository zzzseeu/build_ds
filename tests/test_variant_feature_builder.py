import logging
import unittest

from variant_feature_builder import VariantFeatureBuilder


def make_builder() -> VariantFeatureBuilder:
    builder = VariantFeatureBuilder.__new__(VariantFeatureBuilder)
    builder.logger = logging.getLogger("test.variant_feature_builder")
    return builder


def make_row(position: int, ref: str, alt: str, sample_value: int = 1) -> dict[str, str | int]:
    return {
        "Chromosome": "Chr1",
        "Position": position,
        "Gene": "GeneA",
        "REF": ref,
        "ALT": alt,
        "sample_a": sample_value,
    }


class ApplyVariantsToGeneTest(unittest.TestCase):
    def test_insertion(self) -> None:
        observed = make_builder()._apply_variants_to_gene(
            reference_seq="AACCGGTT",
            gene_start=100,
            variant_rows=[make_row(position=102, ref="C", alt="CTT")],
            sample="sample_a",
        )

        self.assertEqual(observed, "AACTTCGGTT")

    def test_deletion(self) -> None:
        observed = make_builder()._apply_variants_to_gene(
            reference_seq="AACCGGTT",
            gene_start=100,
            variant_rows=[make_row(position=102, ref="CCG", alt="C")],
            sample="sample_a",
        )

        self.assertEqual(observed, "AACGTT")

    def test_substitution(self) -> None:
        observed = make_builder()._apply_variants_to_gene(
            reference_seq="AACCGGTT",
            gene_start=100,
            variant_rows=[make_row(position=104, ref="GG", alt="TA")],
            sample="sample_a",
        )

        self.assertEqual(observed, "AACCTATT")

    def test_overlap_skips_later_variant(self) -> None:
        observed = make_builder()._apply_variants_to_gene(
            reference_seq="AACCGGTT",
            gene_start=100,
            variant_rows=[
                make_row(position=102, ref="CC", alt="T"),
                make_row(position=103, ref="CG", alt="A"),
            ],
            sample="sample_a",
        )

        self.assertEqual(observed, "AATGGTT")

    def test_absent_sample_variant_is_skipped(self) -> None:
        observed = make_builder()._apply_variants_to_gene(
            reference_seq="AACCGGTT",
            gene_start=100,
            variant_rows=[make_row(position=102, ref="C", alt="CTT", sample_value=0)],
            sample="sample_a",
        )

        self.assertEqual(observed, "AACCGGTT")


if __name__ == "__main__":
    unittest.main()
