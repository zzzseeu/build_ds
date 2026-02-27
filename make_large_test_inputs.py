import csv
import random
from pathlib import Path

random.seed(42)

base_dir = Path('/path/to/project/test_inputs')
base_dir.mkdir(parents=True, exist_ok=True)

chroms = [f'Chr{i}' for i in range(1, 11)]
chrom_len = 1000

# 100 genes: 10 genes per chromosome, each gene length 80; total gene-covered sites = 10*10*80=8000
genes = []
qtl_rows = []
for chrom in chroms:
    for gi in range(10):
        start = gi * 80 + 1
        end = start + 79
        gene_id = f'{chrom}_Gene_{gi+1:02d}'
        genes.append((chrom, start, end, gene_id, chrom))
        qtl_rows.append((chrom, start, end, f'Trait_{(gi % 5) + 1}', f'QTL_{gene_id}'))

# FASTA sequences
bases = ['A', 'C', 'G', 'T']
fasta_seqs = {}
for chrom in chroms:
    seq = ''.join(random.choice(bases) for _ in range(chrom_len))
    fasta_seqs[chrom] = seq

# Write FASTA
fasta_path = base_dir / 'genome.fa'
with fasta_path.open('w') as f:
    for chrom in chroms:
        f.write(f'>{chrom}\n')
        seq = fasta_seqs[chrom]
        for i in range(0, len(seq), 80):
            f.write(seq[i:i+80] + '\n')

# Write gene_df: include relation to FASTA via Fasta_id
with (base_dir / 'gene_df.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Chromosome', 'Start', 'End', 'Gene_id', 'Fasta_id'])
    for row in genes:
        w.writerow(row)

# Keep a gene_intervals.csv compatible with previous scripts
with (base_dir / 'gene_intervals.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Chromosome', 'Start', 'End', 'Gene_id'])
    for chrom, start, end, gene_id, _ in genes:
        w.writerow([chrom, start, end, gene_id])

# QTL intervals
with (base_dir / 'qtl_intervals.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Chromosome', 'Start', 'End', 'Trait', 'QTL_name'])
    for row in qtl_rows:
        w.writerow(row)

# Union target U = all gene-covered sites: positions 1..800 on each chromosome => 8000 sites
target_union_sites = []
for chrom in chroms:
    for pos in range(1, 801):
        target_union_sites.append((chrom, pos))

# GWAS: random subset of U (3000 sites)
gwas_sites = random.sample(target_union_sites, 3000)
with (base_dir / 'gwas_sites.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Chromosome', 'Position', 'Trait'])
    for chrom, pos in sorted(gwas_sites, key=lambda x: (x[0], x[1])):
        w.writerow([chrom, pos, f'Trait_{(pos % 5) + 1}'])

# VCF: 10000 sites = 1000 per chromosome
vcf_path = base_dir / 'variants.vcf'
with vcf_path.open('w') as f:
    f.write('##fileformat=VCFv4.2\n')
    f.write('##source=synthetic_large_test\n')
    f.write('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n')
    for chrom in chroms:
        seq = fasta_seqs[chrom]
        for pos in range(1, chrom_len + 1):
            ref = seq[pos - 1]
            alt_candidates = [b for b in bases if b != ref]
            alt = random.choice(alt_candidates)
            f.write(f'{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t.\n')

print('Generated files:')
for p in [
    base_dir / 'gwas_sites.csv',
    base_dir / 'qtl_intervals.csv',
    base_dir / 'variants.vcf',
    base_dir / 'genome.fa',
    base_dir / 'gene_df.csv',
    base_dir / 'gene_intervals.csv',
]:
    print(p)
