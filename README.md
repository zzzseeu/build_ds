# Build DS Toolkit

当前项目以 [site_dataset_builder.py](/Users/seeu/Desktop/Project/build_ds/site_dataset_builder.py) 作为唯一推荐的完整流程入口，用于从 `GWAS/QTL/GFF3/VCF/FASTA` 构建多种样本级特征数据集。

## 当前保留的核心模块

### 1. `site_dataset_builder.py`

这是当前项目的主脚本，负责完成整条数据构建流程，包括：

- 读取并标准化 GWAS、QTL、GFF3、VCF、FASTA
- 根据 `type=union/intersect` 筛选候选位点
- 根据基因区间和 `ext_len` 进一步过滤位点
- 可选根据 `gene_csv_path` 保留指定基因
- 从 VCF 中提取样本基因型并构建 `genotype_012`
- 基于局部序列上下文构建 `variant_effect_matrix`
- 基于样本特异性基因序列构建 `gene_sequence_feature_matrix`
- 可选进一步自动拆分 `train_val/test` 数据集

支持的主输入参数包括：

- `gwas_csv_path`
- `qtl_csv_path`
- `gff3_path`
- `fasta_path`
- `vcf_path`
- `type`

常用扩展参数包括：

- `ext_len`
- `k`
- `gene_csv_path`
- `pvalue_threshold`
- `LOD_threshold`
- `PVE_threshold`
- `embedder_type`
- `model_name_or_path`
- `split_test_ratio`
- `isolated_sample_csv`
- `split_random_state`

### 2. `utils.py`

当前 `utils.py` 只保留基础通用能力，不再承载重复的数据构建流程函数。

主要包括：

- `initLogger`
- `getLogger`
- `standard_chrom`
- `standard_sample_name`
- `parse_gff3_attributes`
- `extract_gff3_feature_intervals`
- `extract_gff3_feature_intervals_gffutils`
- `build_feature_interval_trees`
- `extract_gff3_feature_interval_trees`
- `extract_gff3_feature_interval_trees_gffutils`
- `query_feature_interval_trees`

### 3. `embedder.py`

负责具体模型加载与序列 embedding 计算。

### 4. `embedding.py`

提供 `UnifiedEmbedder` 统一封装，用于在主脚本中按统一接口调用不同 embedder。

### 5. `variant_tsv_embedding_classifier.py`

用于读取 `TSV + VCF + FASTA`：

- 从 TSV 指定的位点中匹配 VCF 变异
- 提取位点上下游 `k` 长度窗口
- 分别构造 `REF` 和 `ALT` 序列
- 对 `REF/ALT` 序列做 embedding
- 以 `VARIANT_TYPE` 作为标签做二分类或多分类建模
- 以 embedding 向量作为特征进行监督学习评估
- 输出 `AUC / PR AUC / F1 / Accuracy` 等指标

典型运行方式：

```bash
python3 /Users/seeu/Desktop/Project/build_ds/variant_tsv_embedding_classifier.py \
  --tsv-path /path/to/sites.tsv \
  --vcf-path /path/to/variants.vcf.gz \
  --fasta-path /path/to/genome.fa \
  --outdir /path/to/variant_tsv_alt_pca_out \
  --k 100 \
  --embedder-type rice8k \
  --model-name-or-path /path/to/embedder_model \
  --device cuda \
  --task-type auto \
  --feature-type alt \
  --model-type logistic_regression \
  --local-files-only
```

主要输出：

- `variant_sequences.tsv`
- `labels.tsv`
- `features.npy`
- `predictions.tsv`
- `metrics.json`
- `confusion_matrix.tsv`
- `classification_report.txt`
- `ref_embeddings.npy`
- `alt_embeddings.npy`
- `alt_minus_ref_embeddings.npy`

## 已清理的重复脚本

以下旧脚本的主要功能已经被 `site_dataset_builder.py` 覆盖，因此已清理：

- `gwas_qtl_variant_extractor.py`
- `variant_feature_builder.py`
- `split_variant_datasets.py`

对应的旧测试也已同步清理。

## 依赖

推荐在生信环境中安装：

```bash
conda activate py-bioinfo
pip install pandas numpy pysam intervaltree transformers loguru pyyaml tqdm
```

如果使用特定 DNA 语言模型，还需要对应模型权重和运行环境。

## 输入数据格式

### GWAS 文件

必需列：

- `Chromosome`
- `Position`
- `Trait`

常见可选列：

- `pvalue`

### QTL 文件

必需列：

- `Chromosome`
- `start_pos`
- `end_pos`
- `Trait`

常见可选列：

- `QTL`
- `LOD`
- `PVE`

### GFF3/GTF 文件

用于提供基因区间注释。

### VCF 文件

用于提取样本基因型信息。

### FASTA 文件

用于提取局部序列和基因参考序列。

## 命令行运行方式

脚本通过 YAML 配置文件运行：

```bash
python3 /Users/seeu/Desktop/Project/build_ds/site_dataset_builder.py \
  --config /Users/seeu/Desktop/Project/build_ds/site_dataset_builder.example.yaml
```

## 配置模板

项目中提供了配置模板：

- [site_dataset_builder.example.yaml](/Users/seeu/Desktop/Project/build_ds/site_dataset_builder.example.yaml)

## 逻辑说明文档

完整逻辑说明见：

- [SITE_DATASET_BUILDER_LOGIC.md](/Users/seeu/Desktop/Project/build_ds/SITE_DATASET_BUILDER_LOGIC.md)

## 典型输出

主流程通常会输出：

- `sites_union_or_intersect.csv`
- `sites_in_gene.csv`
- `sample_name_mapping.csv`
- `genotype_012_site_by_sample.csv`
- `genotype_012.csv`
- `genotype_012.parquet`
- `variant_effect_matrix.csv`
- `variant_effect_matrix.parquet`
- `gene_sequence_feature_matrix.csv`
- `gene_sequence_feature_matrix.parquet`
- `site_dataset_builder.log`

如果配置了数据集拆分，还会额外输出：

- `splits/train_val_genotype_012.csv`
- `splits/train_val_genotype_012.parquet`
- `splits/test_genotype_012.csv`
- `splits/test_genotype_012.parquet`
- `splits/train_val_variant_effect_matrix.csv`
- `splits/train_val_variant_effect_matrix.parquet`
- `splits/test_variant_effect_matrix.csv`
- `splits/test_variant_effect_matrix.parquet`
- `splits/train_val_gene_sequence_feature_matrix.csv`
- `splits/train_val_gene_sequence_feature_matrix.parquet`
- `splits/test_gene_sequence_feature_matrix.csv`
- `splits/test_gene_sequence_feature_matrix.parquet`
- `splits/split_meta.json`
