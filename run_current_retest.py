import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from variant_feature_builder import VariantFeatureBuilder


class MockEmbedder:
    def __init__(self, dim=16):
        self.dim = dim

    def __call__(self, sequence):
        if isinstance(sequence, list):
            return np.vstack([self._one(s) for s in sequence])
        return self._one(sequence)

    def _one(self, s):
        h = hashlib.sha1(s.encode("utf-8")).digest()
        arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        if len(arr) < self.dim:
            arr = np.pad(arr, (0, self.dim - len(arr)), mode="wrap")
        return (arr[:self.dim] / 255.0).astype(np.float32)


outdir = Path('/Users/seeu/Desktop/Project/build_ds/test_outputs/feature_current_retest')
builder = VariantFeatureBuilder(
    variant_df_path='/Users/seeu/Desktop/Project/build_ds/test_outputs/current_retest_union.csv',
    vcf_path='/Users/seeu/Desktop/Project/build_ds/test_inputs/variants.vcf',
    fasta_path='/Users/seeu/Desktop/Project/build_ds/test_inputs/genome.fa',
    outdir=str(outdir),
    embedder=MockEmbedder(dim=16),
    isolated_sample=['S001', 'S002'],
    test_ratio=0.2,
    val_ratio=0.1,
    random_seed=42,
    flank_k=20,
    pca_var_threshold=0.95,
    save_file=True,
    file_format='csv',
)
outputs = builder.run()

print('=== output shapes ===')
for k, v in outputs.items():
    print(k, v.shape)

# automatic checks
print('\n=== checks ===')
assert outputs['genotype_012'].shape[0] == 60, 'sample count should be 60'
assert outputs['split_train'].shape[0] + outputs['split_val'].shape[0] + outputs['split_test'].shape[0] == 60

# check isolated samples are in test
stest = set(outputs['split_test']['sample'])
assert 'S001' in stest and 'S002' in stest, 'isolated samples must be in test'
print('split checks passed')

# check feature naming convention on genotype_012/extseq_raw/distance_x_gt
pat = re.compile(r'^Chr[^:]*:\d+:[ACGTN]>[ACGTN]-.+$')
for name in ['genotype_012', 'extseq_raw', 'distance_x_gt']:
    cols = [c for c in outputs[name].columns if c != 'sample']
    bad = [c for c in cols[:200] if not pat.match(c)]
    assert not bad, f'{name} naming bad examples: {bad[:3]}'
print('naming checks passed')

# check outputs written
required = [
    'genotype_012.csv', 'gene_sequence.csv', 'gene_pca.csv',
    'concat_embedding.csv', 'extseq_raw.csv', 'extseq_pca.csv',
    'distance_x_gt.csv', 'split_train.csv', 'split_val.csv', 'split_test.csv',
]
for fn in required:
    assert (outdir / fn).exists(), f'missing file: {fn}'
print('file checks passed')

# extra: if duplicated (chrom,pos,gene) records exist, columns should reflect record count
src = pd.read_csv('/Users/seeu/Desktop/Project/build_ds/test_outputs/current_retest_union.csv')
n_records = src[['Chromosome', 'Position', 'Gene_id']].drop_duplicates().shape[0]
assert outputs['genotype_012'].shape[1] - 1 == n_records, '012 columns should match record count'
print('record-level checks passed')

print('\nALL CHECKS PASSED')
