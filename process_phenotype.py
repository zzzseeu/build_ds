import pandas as pd
import os
import datetime as dt


def process_trait_data(phenotype_df, trait_name):
    """
    处理表型数据的通用函数

    参数:
        phenotype_df: 原始表型数据DataFrame
        trait_name: 性状名称 (如 'SPY', 'ETN', 'TGP', 'TGW')

    返回:
        处理后的DataFrame，包含one-hot编码的地点信息
    """
    # 筛选包含指定性状的列
    trait_cols = [col for col in phenotype_df.columns if trait_name in col]
    trait_data = phenotype_df[["sample"] + trait_cols]

    # 将宽矩阵转换为长矩阵
    trait_data_long = pd.melt(
        trait_data,
        id_vars=["sample"],
        value_vars=trait_cols,
        var_name=f"{trait_name}_trait",
        value_name="value",
    )

    # 过滤缺失值
    trait_data_long = trait_data_long.dropna(subset=["value"])

    # 拆分性状列
    trait_data_long[["trait", "year", "location"]] = trait_data_long[
        f"{trait_name}_trait"
    ].str.split("_", expand=True)

    # 对同一地点不同年份的value取均值
    trait_data_agg = (
        trait_data_long.groupby(["sample", "location"])["value"].mean().reset_index()
    )

    # 保留三位小数
    trait_data_agg["value"] = trait_data_agg["value"].round(3)

    # 将location列转换为one-hot编码
    trait_data_agg = pd.get_dummies(
        trait_data_agg, columns=["location"], prefix="loc", dtype=int
    )

    return trait_data_agg


# 读取表型数据
phenotype = pd.read_excel("../rawdata/raw_phenotype_20260124.xlsx", sheet_name="triple_data")

# 处理不同性状
traits = ["SPY", "ETN", "TGP", "TGW"]
outdir = "phenotype"
date = dt.datetime.now().strftime("%Y%m%d")
if not os.path.exists(outdir):
    os.makedirs(outdir)
for trait in traits:
    trait_data = process_trait_data(phenotype, trait)
    trait_data.to_csv(f"{outdir}/phenotype_{trait}_processed_{date}.csv", index=False)
    print(f"{trait} 数据处理完成，共 {len(trait_data)} 行")
