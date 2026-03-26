# site_dataset_builder 逻辑说明

## 1. 脚本用途

`site_dataset_builder.py` 用于将 GWAS、QTL、基因注释、VCF、FASTA 和序列 embedding 串联起来，自动构建可直接用于机器学习或深度学习分析的多种数据集。

脚本最终构建 3 类核心数据集：

1. `genotype_012` 数据集
2. `variant_effect_matrix` 数据集
3. `gene_sequence_feature_matrix` 数据集

整个流程由一个类完成：

- `SiteDatasetBuilder`

脚本支持通过 YAML 配置文件从命令行运行。

## 2. 输入文件

核心输入文件如下：

- `gwas_csv_path`
- `qtl_csv_path`
- `gff3_path`
- `fasta_path`
- `vcf_path`

核心控制参数如下：

- `type`
- `ext_len`
- `k`
- `gene_csv_path`
- `pvalue_threshold`
- `LOD_threshold`
- `PVE_threshold`
- `embedder_type`
- `model_name_or_path`
- `device`
- `pooling`
- `local_files_only`
- `embedder_kwargs`

## 3. 输出文件

脚本运行后，通常会生成以下文件：

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

## 4. 染色体标准化规则

所有输入文件中出现的染色体名称，都会统一标准化为 `ChrN` 格式。

例如：

- `1` 会转换为 `Chr1`
- `chr1` 会转换为 `Chr1`
- `chr01` 会转换为 `Chr1`

该规则会应用到以下输入：

- GWAS 文件
- QTL 文件
- VCF 文件
- FASTA 文件
- GFF3 或 GTF 文件

## 5. 样本名标准化规则

所有从 VCF 中读取的样本名，都会统一标准化为 `sample_N` 格式。

例如：

- `1` 会转换为 `sample_1`
- `sample1` 会转换为 `sample_1`
- `Sample_01` 会转换为 `sample_1`

如果标准化后发生重名，脚本会自动追加后缀，保证样本名唯一。

脚本还会输出样本名映射文件：

- `sample_name_mapping.csv`

该文件包含两列：

- `raw_sample`
- `sample`

用于保存原始样本名与标准化样本名之间的映射关系。

## 6. 整体流程概览

整个脚本的处理逻辑可以概括为以下 7 个步骤：

1. 读取并过滤 GWAS 数据
2. 读取并过滤 QTL 数据
3. 基于 GWAS、QTL 和 VCF 筛选候选位点
4. 基于基因区间进一步过滤位点
5. 提取样本基因型并构建 012 数据集
6. 基于局部序列构建位点突变效应数据集
7. 基于样本特异性基因序列构建基因特征数据集

下面对每一步进行详细说明。

## 7. 第一步：读取并过滤 GWAS 数据

脚本首先使用 pandas 读取 GWAS 文件。

GWAS 文件要求至少包含以下列：

- `Chromosome`
- `Position`
- `Trait`

如果文件中包含 `pvalue` 列，则会按以下规则过滤：

- `pvalue < pvalue_threshold`

处理后的 GWAS 数据会被整理成按位点索引的 trait 映射表，供后续快速判断某个位点是否被 GWAS 支持。

## 8. 第二步：读取并过滤 QTL 数据

脚本随后读取 QTL 文件。

QTL 文件要求至少包含以下列：

- `Chromosome`
- `start_pos`
- `end_pos`
- `Trait`

如果文件中包含以下列，则会进一步过滤：

- `LOD > LOD_threshold`
- `PVE > PVE_threshold`

过滤后的 QTL 区间会被构造成按染色体组织的区间树，用于后续快速判断某个位点是否落在 QTL 区间内。

## 9. 第三步：基于 GWAS、QTL 和 VCF 筛选候选位点

脚本会扫描整个 VCF 文件。

对每一个 VCF 位点，执行以下操作：

1. 标准化染色体名称
2. 读取位点位置
3. 查询该位点是否在 GWAS 中出现
4. 查询该位点是否落在 QTL 区间内
5. 汇总该位点对应的 GWAS trait 和 QTL trait

然后根据参数 `type` 决定是否保留：

- 当 `type=intersect` 时，只保留同时被 GWAS 和 QTL 支持的位点
- 当 `type=union` 时，保留被 GWAS 或 QTL 任一方支持的位点

这一阶段输出的位点表包括以下列：

- `Chromosome`
- `Position`
- `qtl_trait`
- `gwas_trait`

脚本内部还会保留：

- `REF`
- `ALT`

这些信息将用于后续序列效应计算。

该阶段保存的文件为：

- `sites_union_or_intersect.csv`

## 10. 第四步：基于基因区间进一步过滤位点

脚本会解析 `gff3_path` 指定的基因注释文件，并构建基因区间树。

参数 `ext_len` 用于对每个基因区间上下游分别扩展指定长度。

随后，对第三步得到的候选位点逐一进行基因区间查询：

- 如果位点落在扩展后的基因区间内，则保留
- 如果一个位点命中多个基因，则保留多行

这一阶段会为位点补充以下注释：

- `gene_id`
- `gene_start`
- `gene_end`

如果提供了 `gene_csv_path`，则脚本会进一步只保留指定基因中的位点。

该阶段保存的文件为：

- `sites_in_gene.csv`

该文件包含以下列：

- `Chromosome`
- `Position`
- `qtl_trait`
- `gwas_trait`
- `gene_id`

## 11. 第五步：构建 012 基因型数据集

在最终位点确定之后，脚本会再次扫描 VCF，提取所有样本在这些位点上的基因型。

基因型编码规则如下：

- `0` 表示参考型或缺失型
- `1` 表示杂合型
- `2` 表示 ALT 等位纯合型或 ALT 剂量为 2

脚本会生成两个版本的基因型数据：

1. 位点为行、样本为列
2. 样本为行、位点为列

其中常用于建模的是第二种格式，也就是：

- 行为样本
- 列为位点
- 位点列名格式为 `ChrN:Pos`

保存的文件包括：

- `genotype_012_site_by_sample.csv`
- `genotype_012.csv`
- `genotype_012.parquet`

## 12. 第六步：构建位点突变效应数据集

对于每一个最终保留的位点，脚本会从 FASTA 中提取其上下游序列。

窗口长度由参数 `k` 控制。

对每个位点，构造两条局部 DNA 序列：

- `REF 序列 = upstream + REF + downstream`
- `ALT 序列 = upstream + ALT + downstream`

随后，使用配置好的 embedder 分别对 REF 序列和 ALT 序列进行编码，并计算二者 embedding 的欧氏距离。

这个欧氏距离定义为该位点的局部突变效应：

- `var_effect`

再将 `var_effect` 与样本的 012 基因型相乘，得到“样本 x 位点”的突变效应矩阵。

含义如下：

- 如果某个样本在该位点的基因型为 `0`，则该位点效应值为 `0`
- 如果基因型为 `1`，则该位点效应值为 `1 × var_effect`
- 如果基因型为 `2`，则该位点效应值为 `2 × var_effect`

输出文件为：

- `variant_effect_matrix.csv`
- `variant_effect_matrix.parquet`

## 13. 第七步：构建基因序列特征数据集

脚本会将最终保留的位点按照 `gene_id` 进行分组。

对于每个基因，执行以下操作：

1. 从 FASTA 中提取参考基因序列
2. 对每个样本，根据该样本在该基因中所有位点上的 012 状态，将 ALT 突变应用到参考序列上
3. 生成该样本的突变后基因序列
4. 对参考序列和样本突变后序列分别做 embedding
5. 计算两者 embedding 的欧氏距离

这个欧氏距离作为该样本在该基因上的序列特征值。

最终得到一个“样本 x 基因”的矩阵：

- 行为样本
- 列为基因
- 值为样本对应基因的序列效应特征

输出文件为：

- `gene_sequence_feature_matrix.csv`
- `gene_sequence_feature_matrix.parquet`

## 14. Embedding 处理方式

脚本中的序列 embedding 通过 `UnifiedEmbedder` 完成。

相关参数包括：

- `embedder_type`
- `model_name_or_path`
- `device`
- `pooling`
- `local_files_only`
- `embedder_kwargs`

为了避免重复编码，相同序列的 embedding 会进行缓存。

缓存目录为：

- `embedding_cache/`

缓存键基于序列的 SHA1 值生成。

## 15. 日志与进度条

脚本在长循环中使用 `tqdm` 展示进度。

日志系统使用 `loguru`，并通过 `utils.py` 中的统一入口进行初始化。

日志记录的关键节点包括：

- 数据读取
- GWAS/QTL 位点筛选
- 基因区间映射
- VCF 基因型提取
- 位点突变效应计算
- 基因序列特征构建
- 输出文件保存

日志文件为：

- `site_dataset_builder.log`

## 16. 命令行运行方式

脚本通过 YAML 配置文件运行，示例如下：

```bash
python3 site_dataset_builder.py --config /path/to/config.yaml
```

项目中提供了一个示例配置文件：

- `site_dataset_builder.example.yaml`

## 17. 最终理解

可以将 `site_dataset_builder.py` 理解为一条完整的数据构建流水线。

它的核心目标是：

1. 用 GWAS 和 QTL 信息定位候选位点
2. 用基因区间限制位点范围
3. 用 VCF 构建样本级基因型矩阵
4. 用 FASTA 和 embedder 量化位点级和基因级序列效应
5. 输出统一染色体命名、统一样本命名的标准化分析数据集

最终得到的数据，既保留了位点层面的变异信息，也保留了基因层面的序列效应信息，便于后续用于统计建模、机器学习和深度学习任务。
