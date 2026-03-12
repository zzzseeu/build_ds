# Build DS Toolkit

用于从 `GWAS/QTL/VCF/FASTA` 构建多种变异特征数据集的脚本集合。

## 模块说明

### 1) `gwas_qtl_variant_extractor.py`
- 作用：
  - 读取 GWAS 位点表与 QTL 区间表；
  - 使用 QTL 区间构建按染色体组织的 interval tree；
  - 在 VCF 位点上计算 GWAS 与 QTL 的 `intersect/union`；
  - 将候选位点固定映射到 `gff3` 给定 feature 区间上；
  - 支持按 `Trait`、`Gene` 与阈值过滤（`pvalue/LOD/PVE`）；
  - 输出候选位点信息表，不直接输出样本基因型矩阵；
  - 坐标统一按 1-based 输出。
- 输入参数（CLI）：
  - 必需：`--gwas_csv_path --qtl_csv_path --type --vcf_path --gff3_path --outprefix`
  - 可选：`--trait --pvalue_threshold --LOD_threshold --PVE_threshold --gff3_feature --ext_len --gene_list --use_gffutils/--no-use_gffutils`
  - `--trait` 支持单个 trait 或逗号分隔多个 trait（例如 `Yield` 或 `Yield,Height`）
  - `--gene_list` 支持逗号分隔多个基因名或基因 ID（例如 `GeneA,GeneB`）
  - `--gff3_feature` 支持单个或逗号分隔多个 feature（默认 `gene`）
  - `--ext_len` 表示区间上下游扩展长度，默认 `500`
  - `--use_gffutils` 默认开启，使用 `gffutils` 解析 `gff3`
- 输出文件：
  - `{outprefix}_{YYYY-MM-DD}.csv`
  - 列为：`Chromosome, Position, Gene, gwas_trait, qtl_trait`
- 输入文件格式：
  - GWAS CSV 必需列：`Chromosome, Position, Trait, pvalue`
  - QTL CSV 必需列：`QTL, Chromosome, LOD, PVE, start_pos, end_pos, Trait`
  - GFF3：用于将候选位点映射到给定 feature 区间上，若一个位点落在多个基因区间内，则输出中保留多行
  - `Chromosome` 会做标准化（如 `1/chr1/chr01/chr_01 -> Chr1`），仅保留数字染色体
  - `Position/start_pos/end_pos` 均按 1-based 处理

### 2) `variant_feature_builder.py`
- 作用：
  - 提供 `GWASQTLGenotypeExtractor`，根据位点表 + VCF 提取所有样本的 `0/1/2` 基因型矩阵 `geno_df`；
  - 提供 `VariantFeatureBuilder`，根据 `geno_df + FASTA` 构建 `geno012_df / gene_seq_df`；
  - 支持 embedding 缓存、序列/embedding 字典持久化、按基因 PCA 降维。
- 主要类：
  - `GWASQTLGenotypeExtractor`
  - `VariantFeatureBuilder`
- 使用方式：
  - 当前模块同时提供 CLI 和 Python API；
  - CLI 子命令：`genotype` 用于导出 `geno_df`，`build` 用于构建特征数据集；
  - `GWASQTLGenotypeExtractor.run()` 返回并可选保存 `geno_df`；
  - `VariantFeatureBuilder.run()` 会一次性构建并保存 3 个数据集；
  - 若需要序列类特征，初始化 `VariantFeatureBuilder` 时必须提供 `embedder` 或 `embedder_type`；CLI 中使用 `--embedder_type + --model_name_or_path`。
- 主要输出文件：
  - `{outprefix}.csv`：`GWASQTLGenotypeExtractor` 导出的 `geno_df`
  - `{outdir}/geno012_df.csv`
  - `{outdir}/gene_seq_df.csv`
  - `{outdir}/dicts/ref_gene_seq_dict.json`
  - `{outdir}/dicts/alt_gene_seq_dict.json`
  - `{outdir}/dicts/alt_gene_seq_embedding_dict.json`
  - `{outdir}/dicts/alt_gene_seq_embedding_pca_dict.json`
  - `{outdir}/pca_models/`（当 `use_pca=True`）
  - `{outdir}/embedding_cache/`

### 3) `embedder.py`
- 作用：具体模型加载实现（Rice8k/NT/Evo2 等）。
- `variant_feature_builder.py` 内已集成 `UnifiedEmbedder` 接口，会调用本文件中的模型类。

### 4) `utils.py`
- 作用：
  - 提供染色体名称标准化函数；
  - 提供 `gff3` 纯文本解析与 `gffutils` 解析；
  - 提供 feature 区间表到区间树的转换；
  - 提供按 `chromosome + position` 查询命中区间的方法。
- 主要函数：
  - `standard_chrom`
  - `parse_gff3_attributes`
  - `extract_gff3_feature_intervals`
  - `extract_gff3_feature_intervals_gffutils`
  - `build_feature_interval_trees`
  - `extract_gff3_feature_interval_trees`
  - `extract_gff3_feature_interval_trees_gffutils`
  - `query_feature_interval_trees`

## 依赖

推荐在 `py-bioinfo` 环境中安装：

```bash
conda activate py-bioinfo
pip install pandas numpy pysam intervaltree scikit-learn transformers
```

如果使用 Rice8k 等模型，还需要对应模型权重与运行环境（GPU、flash attention 等）。

## 输入数据格式

### GWAS 文件（CSV）
- 必需列：`Chromosome, Position, Trait, pvalue`

### QTL 文件（CSV）
- 必需列：`QTL, Chromosome, LOD, PVE, start_pos, end_pos, Trait`

### 基因区间文件（CSV，可选）
- 必需列：`Chromosome, Start, End, Gene_id`

### VariantSelector 输出位点文件（CSV）
- 必需列：`Chromosome, Position, Gene_id, Gene_position`

### VCF
- 坐标按 1-based。
- `GWASQTLGenotypeExtractor` 会读取样本基因型 `GT`，并转为 `0/1/2`。

### FASTA
- 染色体 ID 需与 VCF/位点文件可对应。

## 使用示例

## A. 位点整合（GWAS + QTL）

```bash
conda run -n py-bioinfo python /path/to/project/gwas_qtl_variant_extractor.py \
  --gwas_csv_path /path/to/project/test_inputs/gwas.csv \
  --qtl_csv_path /path/to/project/test_inputs/qtl.csv \
  --vcf_path /path/to/project/test_inputs/variants.vcf \
  --gff3_path /path/to/project/test_inputs/annotation.gff3 \
  --type union \
  --outprefix /path/to/project/test_outputs/demo_sites \
  --trait Yield,Height \
  --pvalue_threshold 1e-6 \
  --LOD_threshold 2.5 \
  --PVE_threshold 10 \
  --gff3_feature gene \
  --ext_len 500 \
  --gene_list GeneA,GeneB \
  --use_gffutils
```

输出示例文件：
- `/path/to/project/test_outputs/demo_sites_YYYY-MM-DD.csv`

说明：
- 输出结果是候选位点映射到基因区间后的结果，列为 `Chromosome, Position, Gene, gwas_trait, qtl_trait`
- 若同一位点同时命中多个基因区间，则输出多行

## B. 提取 `geno_df`（Python API）

```python
from variant_feature_builder import GWASQTLGenotypeExtractor

geno_extractor = GWASQTLGenotypeExtractor(
    vcf_path="/path/to/project/test_inputs/variants.vcf",
    site_df_path="/path/to/project/test_outputs/demo_sites_YYYY-MM-DD.csv",
    outprefix="/path/to/project/test_outputs/demo_geno_df",
)

geno_df = geno_extractor.run()
print(geno_df.head())
```

输出示例文件：
- `/path/to/project/test_outputs/demo_geno_df.csv`

说明：
- `geno_df` 前 5 列固定为：
  - `Chromosome`
  - `Position`
  - `Gene`
  - `gwas_trait`
  - `qtl_trait`
- 后续列全部为标准化样本名（如 `sample_1`）对应的 `0/1/2` 基因型

## C. 提取 `geno_df`（CLI）

```bash
conda run -n py-bioinfo python /path/to/project/variant_feature_builder.py genotype \
  --vcf_path /path/to/project/test_inputs/variants.vcf \
  --site_df_path /path/to/project/test_outputs/demo_sites_YYYY-MM-DD.csv \
  --outprefix /path/to/project/test_outputs/demo_geno_df
```

输出示例文件：
- `/path/to/project/test_outputs/demo_geno_df.csv`

## D. 构建特征数据集（CLI）

```bash
conda run -n py-bioinfo python /path/to/project/variant_feature_builder.py build \
  --geno_df_path /path/to/project/test_outputs/demo_geno_df.csv \
  --fasta_path /path/to/project/test_inputs/genome.fa \
  --outdir /path/to/project/test_outputs/feature_demo_cli \
  --embedder_type rice8k \
  --model_name_or_path /path/to/models/rice_1B_stage2_8k_hf \
  --device cuda \
  --pooling mean \
  --embedder_kwargs '{"torch_dtype":"bfloat16","use_flash_attention":true}' \
  --use_pca \
  --pca_var_threshold 0.95
```

说明：
- `build` 子命令会一次性生成 `geno012_df.csv`、`gene_seq_df.csv`
- `--embedder_kwargs` 需要传 JSON 对象字符串
- 如模型已经在本地，保留默认 `--local_files_only` 即可；若允许远程拉取，可添加 `--no-local_files_only`

## E. 构建特征数据集（Python API）

也可以直接通过类初始化传参并调用 `run()`：

```python
from variant_feature_builder import VariantFeatureBuilder

builder = VariantFeatureBuilder(
    geno_df_path="/path/to/project/test_outputs/demo_geno_df.csv",
    fasta_path="/path/to/project/test_inputs/genome.fa",
    outdir="/path/to/project/test_outputs/feature_demo_api",
    embedder_type="rice8k",
    model_name_or_path="/path/to/models/rice_1B_stage2_8k_hf",
    device="cuda",
    pooling="mean",
    local_files_only=True,
    embedder_kwargs={
        "torch_dtype": "bfloat16",
        "use_flash_attention": True,
    },
    use_pca=True,
    pca_var_threshold=0.95,
)

outputs = builder.run()
print(outputs["geno012_df"].shape)
print(outputs["gene_seq_df"].shape)
```

说明：
- `VariantFeatureBuilder` 当前构建 2 类数据集：
  - `geno012_df`：行为样本，列为去重后的 `ChrN:Pos`
  - `gene_seq_df`：行为样本，列为基因突变序列 embedding 或 PCA 特征
- `run()` 返回一个字典，键固定为：`geno012_df`、`gene_seq_df`
- 会额外保存：
  - `dicts/ref_gene_seq_dict.json`
  - `dicts/alt_gene_seq_dict.json`
  - `dicts/alt_gene_seq_embedding_dict.json`
  - `dicts/alt_gene_seq_embedding_pca_dict.json`
  - `pca_models/` 下的每个基因 PCA 模型文件（若 `use_pca=True`）
  - `embedding_cache/` 下按序列 SHA1 命名的 `.npy` 缓存文件
- 如果你已经有自定义 embedding 模型，也可以直接传入 `embedder=<callable>`，其输入应为单条 DNA 序列字符串，输出应为 1D 向量或形如 `(1, D)` 的数组

## F. 两步流程示例

```python
from gwas_qtl_variant_extractor import GWASQTLVariantExtractor
from variant_feature_builder import GWASQTLGenotypeExtractor, VariantFeatureBuilder

# 第一步：提取候选位点信息
site_df = GWASQTLVariantExtractor(
    gwas_csv_path="/path/to/project/test_inputs/gwas.csv",
    qtl_csv_path="/path/to/project/test_inputs/qtl.csv",
    type="union",
    vcf_path="/path/to/project/test_inputs/variants.vcf",
    gff3_path="/path/to/project/test_inputs/annotation.gff3",
    outprefix="/path/to/project/test_outputs/demo_sites",
    trait="Yield",
    use_gffutils=True,
).run()

# 第二步：从位点信息提取 geno_df
geno_df = GWASQTLGenotypeExtractor(
    vcf_path="/path/to/project/test_inputs/variants.vcf",
    site_df=site_df,
    outprefix="/path/to/project/test_outputs/demo_geno_df",
).run()

# 第三步：基于 geno_df 构建特征数据集
outputs = VariantFeatureBuilder(
    geno_df_path="/path/to/project/test_outputs/demo_geno_df.csv",
    fasta_path="/path/to/project/test_inputs/genome.fa",
    outdir="/path/to/project/test_outputs/feature_demo",
    embedder_type="rice8k",
    model_name_or_path="/path/to/models/rice_1B_stage2_8k_hf",
).run()

print(outputs["geno012_df"].head())
```

## G. `utils.py` 使用示例

### 染色体标准化

```python
from utils import standard_chrom

print(standard_chrom("1"))       # Chr1
print(standard_chrom("chr01"))   # Chr1
print(standard_chrom("chr_02"))  # Chr2
```

### 纯文本解析 GFF3 feature 区间

```python
from utils import extract_gff3_feature_intervals

gene_df = extract_gff3_feature_intervals(
    gff3_path="/path/to/project/test_inputs/annotation.gff3",
    feature=["gene", "exon"],
)
print(gene_df.head())
```

### 使用 `gffutils` 解析 GFF3 feature 区间

```python
from utils import extract_gff3_feature_intervals_gffutils

gene_df = extract_gff3_feature_intervals_gffutils(
    gff3_path="/path/to/project/test_inputs/annotation.gff3",
    feature="gene",
)
print(gene_df.head())
```

### 构建 feature 区间树并查询位点

```python
from utils import (
    extract_gff3_feature_interval_trees_gffutils,
    query_feature_interval_trees,
)

trees = extract_gff3_feature_interval_trees_gffutils(
    gff3_path="/path/to/project/test_inputs/annotation.gff3",
    feature="gene",
    ext_len=500,
)

hits = query_feature_interval_trees(
    interval_trees=trees,
    chromosome="chr01",
    position=123456,
)
print(hits)
```

## 输出列命名约定

`variant_feature_builder.py` 的输出列命名约定如下：

- `geno012_df`：`sample` + `ChrN:Pos`
- `gene_seq_df`：`sample` + `Gene_embed_i` 或 `Gene_PCn`

示例：
- `Chr1:100`
- `Chr1:100_PC1`
- `LOC_Os01g01010_PC2`

## 常见问题

### 1) 模型路径报 HuggingFace repo id 错误
- 原因：本地路径不可达或被识别为 repo id。
- 处理：检查模型目录是否存在、权限是否可读，或改为正确的 HF repo 名。

### 2) VCF 无样本列导致 012 特征异常
- 需要包含 `FORMAT` 和样本 `GT` 列。

### 3) 染色体命名不一致（如 `chr1`/`Chr1`/`1`）
- 建议先通过位点提取脚本的标准化流程统一，再进入特征构建。

## 文件清单（当前项目）
- `/path/to/project/gwas_qtl_variant_extractor.py`
- `/path/to/project/variant_feature_builder.py`
- `/path/to/project/embedder.py`
- `/path/to/project/test_inputs/`
- `/path/to/project/test_outputs/`
