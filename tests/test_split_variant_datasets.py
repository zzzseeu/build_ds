import unittest

import pandas as pd

from split_variant_datasets import (
    _align_by_sample,
    _build_split_sample_lists,
    _filter_gene_feature_columns,
    _filter_genotype_columns,
)


class SplitVariantDatasetsTest(unittest.TestCase):
    def test_gene_filter_applies_to_both_modalities(self) -> None:
        genotype_df = pd.DataFrame(
            {
                "sample": ["S1", "S2"],
                "Chr1:10": [0, 1],
                "Chr1:20": [1, 0],
                "Chr2:30": [2, 1],
            }
        )
        gene_feature_df = pd.DataFrame(
            {
                "sample": ["S1", "S2"],
                "GeneA-PC1": [0.1, 0.2],
                "GeneB-PC1": [0.3, 0.4],
            }
        )
        site_df = pd.DataFrame(
            {
                "Chromosome": ["Chr1", "Chr1", "Chr2"],
                "Position": [10, 20, 30],
                "Gene": ["GeneA", "GeneA", "GeneB"],
            }
        )

        filtered_genotype = _filter_genotype_columns(genotype_df, site_df, ["GeneA"])
        filtered_feature = _filter_gene_feature_columns(gene_feature_df, ["GeneA"])

        self.assertEqual(filtered_genotype.columns.tolist(), ["sample", "Chr1:10", "Chr1:20"])
        self.assertEqual(filtered_feature.columns.tolist(), ["sample", "GeneA-PC1"])

    def test_isolated_samples_are_forced_into_test_set(self) -> None:
        train_val, test = _build_split_sample_lists(
            sample_list=["S1", "S2", "S3", "S4", "S5"],
            test_ratio=0.2,
            isolated_sample_list=["S4", "S5"],
            random_state=42,
        )

        self.assertIn("S4", test)
        self.assertIn("S5", test)
        self.assertEqual(sorted(train_val + test), ["S1", "S2", "S3", "S4", "S5"])

    def test_align_by_sample_uses_shared_order(self) -> None:
        genotype_df = pd.DataFrame({"sample": ["S2", "S1"], "Chr1:10": [1, 0]})
        gene_feature_df = pd.DataFrame({"sample": ["S1", "S2", "S3"], "GeneA-PC1": [0.1, 0.2, 0.3]})

        aligned_genotype, aligned_feature = _align_by_sample(genotype_df, gene_feature_df)

        self.assertEqual(aligned_genotype["sample"].tolist(), ["S2", "S1"])
        self.assertEqual(aligned_feature["sample"].tolist(), ["S2", "S1"])


if __name__ == "__main__":
    unittest.main()
