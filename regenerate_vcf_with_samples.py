import random
from pathlib import Path

random.seed(123)
base = Path('/path/to/project/test_inputs')
fa = base / 'genome.fa'
vcf = base / 'variants.vcf'

chrom_seqs = {}
cur = None
buf = []
with fa.open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if cur is not None:
                chrom_seqs[cur] = ''.join(buf)
            cur = line[1:]
            buf = []
        else:
            buf.append(line)
    if cur is not None:
        chrom_seqs[cur] = ''.join(buf)

samples = [f'S{i:03d}' for i in range(1, 61)]
bases = ['A','C','G','T']

with vcf.open('w') as f:
    f.write('##fileformat=VCFv4.2\n')
    f.write('##source=synthetic_large_test\n')
    for chrom, seq in chrom_seqs.items():
        f.write(f'##contig=<ID={chrom},length={len(seq)}>\n')
    f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
    f.write('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t' + '\t'.join(samples) + '\n')

    for chrom in sorted(chrom_seqs.keys(), key=lambda x: int(x.replace('Chr',''))):
        seq = chrom_seqs[chrom]
        for pos in range(1, len(seq)+1):
            ref = seq[pos-1]
            alt = random.choice([b for b in bases if b != ref])
            # slight position-dependent genotype frequency
            p = (pos % 10) / 50.0
            gts = []
            for _ in samples:
                r = random.random()
                if r < 0.72 - p:
                    gt = '0/0'
                elif r < 0.94:
                    gt = '0/1'
                else:
                    gt = '1/1'
                gts.append(gt)
            row = [chrom, str(pos), '.', ref, alt, '.', 'PASS', '.', 'GT'] + gts
            f.write('\t'.join(row) + '\n')

print('Wrote', vcf)
print('samples=', len(samples), 'sites=', sum(len(v) for v in chrom_seqs.values()))
