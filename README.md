# Build DS Toolkit

用于从 `GWAS/QTL/VCF/FASTA` 构建多种变异特征数据集的脚本集合。

## 模块说明

### 1) `gwas_qtl_variant_extractor.py`
- 作用：
  - 读取 GWAS 位点和 QTL 区间；
  - 根据 `intersection/union` 与 VCF 位点整合；
  - 可选按基因区间过滤；
  - 输出标准位点表（1-based）。
- 核心输出列：
  - `Chromosome`
  - `Position`
  - `Gene_id`
  - `Gene_position`

### 2) `variant_feature_builder.py`
- 作用：
  - 读取位点表 + VCF + FASTA；
  - 构建多类型特征矩阵；
  - 支持 embedding 缓存、PCA、数据集切分。
- 核心输出：
  - `genotype_012`
  - `gene_sequence`
  - `gene_pca`
  - `concat_embedding`
  - `extseq_raw`
  - `extseq_pca`
  - `distance_x_gt`
  - `split_train/split_val/split_test`
  - `meta.json`

### 3) `embedder.py`
- 作用：具体模型加载实现（Rice8k/NT/Evo2 等）。
- `variant_feature_builder.py` 内已集成 `UnifiedEmbedder` 接口，会调用本文件中的模型类。

## 依赖

推荐在 `py-bioinfo` 环境中安装：

```bash
conda activate py-bioinfo
pip install pandas numpy pysam intervaltree scikit-learn transformers
```

如果使用 Rice8k 等模型，还需要对应模型权重与运行环境（GPU、flash attention 等）。

## 输入数据格式

### GWAS 文件（CSV）
- 必需列：`Chromosome, Position, Trait`

### QTL 文件（CSV）
- 必需列：`Chromosome, Start, End, Trait, QTL_name`

### 基因区间文件（CSV，可选）
- 必需列：`Chromosome, Start, End, Gene_id`

### VariantSelector 输出位点文件（CSV）
- 必需列：`Chromosome, Position, Gene_id, Gene_position`

### VCF
- 坐标按 1-based。
- `variant_feature_builder.py` 会读取样本基因型 `GT`，并转为 `0/1/2`。

### FASTA
- 染色体 ID 需与 VCF/位点文件可对应。

## 使用示例

## A. 位点整合与过滤（GWAS + QTL）

```bash
conda run -n py-bioinfo python /path/to/project/gwas_qtl_variant_extractor.py \
  --gwas_path /path/to/project/test_inputs/gwas_sites.csv \
  --qtl_path /path/to/project/test_inputs/qtl_intervals.csv \
  --vcf_path /path/to/project/test_inputs/variants.vcf \
  --type union \
  --gene_interval_path /path/to/project/test_inputs/gene_intervals.csv \
  --ext_length 0 \
  --outdir /path/to/project/test_outputs \
  --out_prefix demo
```

输出示例文件：
- `/path/to/project/test_outputs/demo_union.csv`

## B. 构建特征数据集（CLI）

```bash
conda run -n py-bioinfo python /path/to/project/variant_feature_builder.py \
  --variant_df_path /path/to/project/test_outputs/demo_union.csv \
  --vcf_path /path/to/project/test_inputs/variants.vcf \
  --fasta_path /path/to/project/test_inputs/genome.fa \
  --outdir /path/to/project/test_outputs/feature_demo \
  --model_name_or_path /path/to/models/rice_1B_stage2_8k_hf \
  --device cuda \
  --torch_dtype bfloat16 \
  --use_flash_attention \
  --pooling mean \
  --test_ratio 0.2 \
  --val_ratio 0.1 \
  --random_seed 42 \
  --flank_k 20 \
  --pca_var_threshold 0.95 \
  --isolated_sample S001 S002 \
  --file_format csv
```

说明：
- 若 `--model_name_or_path` 不可用，可在 Python API 中传入自定义 mock embedder 进行流程验证。

## C. 构建特征数据集（Python API）

```python
import torch
from variant_feature_builder import VariantFeatureBuilder, UnifiedEmbedder

embedder = UnifiedEmbedder(
    "rice8k",
    model_name_or_path="/path/to/models/rice_1B_stage2_8k_hf",
    device="cuda",
    torch_dtype=torch.bfloat16,
    use_flash_attention=True,
    pooling="mean",
)

builder = VariantFeatureBuilder(
    variant_df_path="/path/to/project/test_outputs/demo_union.csv",
    vcf_path="/path/to/project/test_inputs/variants.vcf",
    fasta_path="/path/to/project/test_inputs/genome.fa",
    outdir="/path/to/project/test_outputs/feature_demo_api",
    embedder=embedder,
    isolated_sample=["S001", "S002"],
    test_ratio=0.2,
    val_ratio=0.1,
    random_seed=42,
    flank_k=20,
    pca_var_threshold=0.95,
    save_file=True,
    file_format="csv",
)

outputs = builder.run()
print(outputs["genotype_012"].shape)
print(outputs["gene_pca"].shape)
```

## 输出列命名约定

位点相关特征列使用统一命名：

`Chr:Position:Ref>Alt-gene_id`

示例：
- `Chr1:100:A>G-GeneX`
- `Chr2:350:C>T-GeneY_PC1`（PCA 列会追加 `_PCn`）

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
