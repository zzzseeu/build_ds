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
  - 提供单一 `VariantFeatureBuilder` 类，直接承接位点 CSV + VCF + FASTA；
  - 先构建 `geno_df` 和 `genotype_012`；
  - 再逐基因生成样本特异序列，逐基因做 embedding 与 PCA；
  - 最终拼接所有基因 PCA block，得到 `gene_feature_matrix`；
  - 支持 embedding 缓存、低内存分块写出和按基因 PCA 降维。
- 主要类：
  - `VariantFeatureBuilder`
- 使用方式：
  - 当前模块同时提供 CLI 和 Python API；
  - 使用单个 `VariantFeatureBuilder.run()` 完成全流程；
  - CLI 也改为单条命令直接完成全流程；
  - 若需要序列类特征，初始化 `VariantFeatureBuilder` 时必须提供 `embedder` 或 `embedder_type`；CLI 中使用 `--embedder_type + --model_name_or_path`；
  - `run()` 返回的是输出文件路径与矩阵形状的轻量字典，不返回完整大矩阵。
- 主要输出文件：
  - `{outdir}/geno_df.csv`
  - `{outdir}/genotype_012.csv`
  - `{outdir}/genotype_012.parquet`
  - `{outdir}/gene_feature_matrix.csv`
  - `{outdir}/gene_feature_matrix.parquet`
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
- `VariantFeatureBuilder` 会读取样本基因型 `GT`，并转为 `0/1/2`。

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

## B. 全流程构建（CLI）

```bash
conda run -n py-bioinfo python /path/to/project/variant_feature_builder.py \
  --site_df_path /path/to/project/test_outputs/demo_sites_YYYY-MM-DD.csv \
  --vcf_path /path/to/project/test_inputs/variants.vcf \
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
- 该命令会一次性生成 `geno_df`、`genotype_012` 和 `gene_feature_matrix`
- `gene_feature_matrix` 的构建逻辑是：逐基因生成样本特异序列，逐基因做 embedding，逐基因做 PCA，再拼接所有基因的 PCA block
- `--embedder_kwargs` 需要传 JSON 对象字符串
- 如模型已经在本地，保留默认 `--local_files_only` 即可；若允许远程拉取，可添加 `--no-local_files_only`

输出示例文件：
- `/path/to/project/test_outputs/feature_demo_cli/geno_df.csv`
- `/path/to/project/test_outputs/feature_demo_cli/genotype_012.csv`
- `/path/to/project/test_outputs/feature_demo_cli/genotype_012.parquet`
- `/path/to/project/test_outputs/feature_demo_cli/gene_feature_matrix.csv`
- `/path/to/project/test_outputs/feature_demo_cli/gene_feature_matrix.parquet`
- `/path/to/project/test_outputs/feature_demo_cli/pca_models/`
- `/path/to/project/test_outputs/feature_demo_cli/embedding_cache/`

## C. 全流程构建（Python API）

也可以直接通过类初始化传参并调用 `run()`：

```python
from variant_feature_builder import VariantFeatureBuilder

builder = VariantFeatureBuilder(
    site_df_path="/path/to/project/test_outputs/demo_sites_YYYY-MM-DD.csv",
    vcf_path="/path/to/project/test_inputs/variants.vcf",
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
print(outputs["genotype_012"]["shape"])
print(outputs["genotype_012"]["csv_path"])
print(outputs["gene_feature_matrix"]["shape"])
print(outputs["gene_feature_matrix"]["parquet_path"])
```

说明：
- `VariantFeatureBuilder` 当前会串联构建 3 类结果：
  - `geno_df`：位点 x 样本基因型矩阵，含位点元信息，中间结果会保存为文件
  - `genotype_012`：行为样本，列为去重后的 `ChrN:Pos`
  - `gene_feature_matrix`：行为样本，列为逐基因 PCA 后拼接的特征
- `run()` 返回一个轻量字典，键固定为：`genotype_012`、`gene_feature_matrix`
- 返回值中保存的是输出文件路径与矩阵形状，不是完整 DataFrame
- 会额外保存：
  - `pca_models/` 下的每个基因 PCA 模型文件（若 `use_pca=True`）
  - `embedding_cache/` 下按序列 SHA1 命名的 `.npy` 缓存文件
- 如果你已经有自定义 embedding 模型，也可以直接传入 `embedder=<callable>`，其输入应为单条 DNA 序列字符串，输出应为 1D 向量或形如 `(1, D)` 的数组

## D. 两步流程示例

```python
from gwas_qtl_variant_extractor import GWASQTLVariantExtractor
from variant_feature_builder import VariantFeatureBuilder

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

# 第二步：直接从位点信息构建全部下游矩阵
outputs = VariantFeatureBuilder(
    site_df=site_df,
    vcf_path="/path/to/project/test_inputs/variants.vcf",
    fasta_path="/path/to/project/test_inputs/genome.fa",
    outdir="/path/to/project/test_outputs/feature_demo",
    embedder_type="rice8k",
    model_name_or_path="/path/to/models/rice_1B_stage2_8k_hf",
).run()

print(outputs["gene_feature_matrix"]["csv_path"])
```

## E. `utils.py` 使用示例

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

- `geno_df`：前 9 列为 `Chromosome, Position, Gene, gene_start, gene_end, REF, ALT, qtl_trait, gwas_trait`
- `genotype_012`：`sample` + `ChrN:Pos`
- `gene_feature_matrix`：`sample` + `Gene-embed-i` 或 `Gene-PCn`

示例：
- `REF`
- `ALT`
- `Chr1:100`
- `LOC_Os01g01010-PC1`

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
