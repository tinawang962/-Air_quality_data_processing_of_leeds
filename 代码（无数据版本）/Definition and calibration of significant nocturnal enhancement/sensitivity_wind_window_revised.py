# -*- coding: utf-8 -*-
"""
敏感性检验：风速时间窗口（晚间18:00–23:00 vs 全天平均）——修正版
================================================================

目的
----
在完全相同的日期样本上，比较：
1. 主分析风速：Bramham 18:00–23:00 平均风速；
2. 敏感性风速：Bramham 全天24小时平均风速。

固定不变
--------
- HDD来源：Bradford；
- 响应变量：每日有效站点中出现 SignificantEveningPeak 的站点数；
- SignificantEveningPeak 阈值：14.87 µg/m³；
- 事件日排除规则：主分析版本；
- Winter1 为固定效应参考组；
- 使用日层面分组二项 GLM；
- 使用 Pearson chi-square / residual df 调整过度离散。

主要输出
--------
1. sensitivity_wind_window_all_variables.csv
   保存所有模型中每个变量的 beta、调整后SE、OR、95%CI、调整后p值等。

2. sensitivity_wind_window_hdd_summary.csv
   仅提取 HDD 结果，便于正文比较。

3. sensitivity_wind_window_wind_summary.csv
   比较晚间风速和全天风速本身的效应。

4. sensitivity_wind_window_diagnostics.csv
   保存样本量、日期范围、相关性、VIF等诊断结果。

5. day_level_with_allday_wind_matched.csv
   保存晚间/全天风速严格共同样本。

注意
----
- 全天风速至少需要18个有效小时；
- 晚间风速沿用主数据集中已经计算好的变量；
- 两种风速模型使用完全相同的 matched sample；
- 不根据p值选择主模型。
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")


# ============================================================
# 配置区
# ============================================================

DATA_DIR = Path(r"C:\Users\17793\Desktop\毕业设计材料")
OUTPUT_DIR = DATA_DIR

BRAMHAM_HOURLY_PATH = DATA_DIR / "bramham_hourly_temp_2022_2025_FINAL.csv"
COMPLETE_CASE_PATH = DATA_DIR / "day_level_complete_case.csv"

MIN_HOURS_FOR_ALLDAY_WIND = 18

MATCHED_DATA_OUT = OUTPUT_DIR / "day_level_with_allday_wind_matched.csv"
ALL_VARIABLES_OUT = OUTPUT_DIR / "sensitivity_wind_window_all_variables.csv"
HDD_SUMMARY_OUT = OUTPUT_DIR / "sensitivity_wind_window_hdd_summary.csv"
WIND_SUMMARY_OUT = OUTPUT_DIR / "sensitivity_wind_window_wind_summary.csv"
DIAGNOSTICS_OUT = OUTPUT_DIR / "sensitivity_wind_window_diagnostics.csv"


# ============================================================
# 工具函数
# ============================================================

def require_columns(df, columns, label):
    """检查必要字段。"""
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{label} 缺少必要字段: {missing}")


def two_sided_normal_p(z):
    """稳定计算双侧正态近似p值。"""
    return 2 * stats.norm.sf(abs(z))


def calculate_vif(data, columns):
    """
    对指定自变量计算VIF。
    自动加入常数项，但不返回常数项VIF。
    """
    X = data[columns].astype(float).copy()
    X = sm.add_constant(X)

    rows = []
    for idx, col in enumerate(X.columns):
        if col == "const":
            continue
        rows.append({
            "variable": col,
            "VIF": variance_inflation_factor(X.values, idx)
        })

    return pd.DataFrame(rows)


def fit_grouped_binomial(data, predictors, label):
    """
    拟合日层面分组二项GLM，并对所有系数进行过度离散调整。

    因变量：
      successes = n_significant
      failures  = n_valid_stations - n_significant
    """
    required = [
        "n_significant",
        "n_valid_stations",
        *predictors
    ]
    require_columns(data, required, label)

    y = np.column_stack([
        data["n_significant"].astype(float),
        (
            data["n_valid_stations"]
            - data["n_significant"]
        ).astype(float)
    ])

    X = data[predictors].astype(float).copy()
    X = sm.add_constant(X, has_constant="add")

    model = sm.GLM(
        y,
        X,
        family=sm.families.Binomial()
    ).fit()

    dispersion = model.pearson_chi2 / model.df_resid

    se_adjusted = model.bse * np.sqrt(dispersion)
    z_adjusted = model.params / se_adjusted
    p_adjusted = pd.Series(
        [two_sided_normal_p(z) for z in z_adjusted],
        index=model.params.index
    )

    rows = []

    for variable in model.params.index:
        beta = model.params[variable]
        se_adj = se_adjusted[variable]

        rows.append({
            "model": label,
            "variable": variable,
            "beta": beta,
            "SE_adjusted": se_adj,
            "z_adjusted": z_adjusted[variable],
            "OR": np.exp(beta),
            "CI95_low": np.exp(beta - 1.96 * se_adj),
            "CI95_high": np.exp(beta + 1.96 * se_adj),
            "p_adjusted": p_adjusted[variable],
            "dispersion": dispersion,
            "n_days": len(data),
            "df_resid": model.df_resid,
            "pearson_chi2": model.pearson_chi2,
            "deviance": model.deviance,
            "AIC_unadjusted": model.aic
        })

    return model, pd.DataFrame(rows)


def print_hdd_table(results, title):
    """打印HDD结果对比表。"""
    hdd = results[results["variable"] == "hdd"].copy()

    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    print(
        f"{'Model':<38}"
        f"{'HDD beta':>11}"
        f"{'OR':>10}"
        f"{'95% CI':>23}"
        f"{'adj. p':>12}"
        f"{'dispersion':>14}"
        f"{'n':>7}"
    )

    for _, row in hdd.iterrows():
        ci = f"{row['CI95_low']:.4f}–{row['CI95_high']:.4f}"
        print(
            f"{row['model']:<38}"
            f"{row['beta']:>11.4f}"
            f"{row['OR']:>10.4f}"
            f"{ci:>23}"
            f"{row['p_adjusted']:>12.4f}"
            f"{row['dispersion']:>14.4f}"
            f"{int(row['n_days']):>7}"
        )


# ============================================================
# 第一步：读取Bramham逐小时数据，计算全天平均风速
# ============================================================

print("=" * 78)
print("第一步：读取Bramham小时数据并计算全天平均风速")
print("=" * 78)

bramham = pd.read_csv(BRAMHAM_HOURLY_PATH)

require_columns(
    bramham,
    ["datetime", "wind_speed_knots"],
    "Bramham小时文件"
)

bramham["datetime"] = pd.to_datetime(
    bramham["datetime"],
    errors="coerce"
)

bramham["wind_speed_knots"] = pd.to_numeric(
    bramham["wind_speed_knots"],
    errors="coerce"
)

n_bad_datetime = bramham["datetime"].isna().sum()
print(f"无法解析datetime的行数: {n_bad_datetime}")

bramham = bramham.dropna(subset=["datetime"]).copy()

n_duplicate = bramham["datetime"].duplicated().sum()
print(f"重复时间戳数量: {n_duplicate}")

if n_duplicate > 0:
    bramham = (
        bramham.sort_values("datetime")
        .drop_duplicates(subset="datetime", keep="first")
        .copy()
    )

n_negative_wind = (bramham["wind_speed_knots"] < 0).sum()
print(f"负风速记录数: {n_negative_wind}")

bramham.loc[
    bramham["wind_speed_knots"] < 0,
    "wind_speed_knots"
] = np.nan

bramham["date"] = bramham["datetime"].dt.normalize()

allday_wind = (
    bramham.groupby("date")["wind_speed_knots"]
    .agg(["mean", "count"])
    .reset_index()
)

allday_wind.columns = [
    "date",
    "wind_speed_knots_allday",
    "n_hours_wind_allday"
]

allday_wind = allday_wind[
    allday_wind["n_hours_wind_allday"]
    >= MIN_HOURS_FOR_ALLDAY_WIND
].copy()

allday_wind["wind_speed_ms_allday"] = (
    allday_wind["wind_speed_knots_allday"] * 0.514444
)

print(
    f"全天风速合格天数（>={MIN_HOURS_FOR_ALLDAY_WIND}小时）: "
    f"{len(allday_wind)}"
)
print(
    f"全天风速日期范围: "
    f"{allday_wind['date'].min().date()} ~ "
    f"{allday_wind['date'].max().date()}"
)


# ============================================================
# 第二步：读取冻结的521天主数据并合并全天风速
# ============================================================

print("\n" + "=" * 78)
print("第二步：构建晚间/全天风速严格共同样本")
print("=" * 78)

d = pd.read_csv(COMPLETE_CASE_PATH)

required_main_columns = [
    "date",
    "hdd_bradford",
    "wind_speed_ms_evening",
    "winter",
    "n_valid_stations",
    "n_significant"
]
require_columns(d, required_main_columns, "主分析日层面数据")

d["date"] = pd.to_datetime(d["date"], errors="coerce")

if d["date"].isna().any():
    raise ValueError("主分析数据中存在无法解析的date")

if d["date"].duplicated().any():
    duplicated_dates = d.loc[
        d["date"].duplicated(keep=False),
        "date"
    ].sort_values()
    raise ValueError(
        "主分析日层面数据存在重复日期，例如: "
        f"{duplicated_dates.head().tolist()}"
    )

print(f"冻结主数据天数: {len(d)}")

d = d.merge(
    allday_wind[
        [
            "date",
            "wind_speed_ms_allday",
            "n_hours_wind_allday"
        ]
    ],
    on="date",
    how="left",
    validate="one_to_one"
)

missing_allday = d["wind_speed_ms_allday"].isna()
excluded_dates = d.loc[
    missing_allday,
    ["date"]
].copy()

print(f"全天风速缺失天数: {missing_allday.sum()}")

if len(excluded_dates) > 0:
    print("因全天风速缺失而排除的日期:")
    print(excluded_dates.to_string(index=False))

d_common = d.dropna(
    subset=[
        "hdd_bradford",
        "wind_speed_ms_evening",
        "wind_speed_ms_allday",
        "winter",
        "n_valid_stations",
        "n_significant"
    ]
).copy()

# 明确Winter1为参考组
d_common["winter"] = pd.Categorical(
    d_common["winter"],
    categories=["Winter1", "Winter2", "Winter3"],
    ordered=True
)

# 数据结构断言
assert d_common[
    [
        "hdd_bradford",
        "wind_speed_ms_evening",
        "wind_speed_ms_allday",
        "winter",
        "n_valid_stations",
        "n_significant"
    ]
].notna().all().all(), "共同样本中仍存在必要变量缺失"

assert (
    d_common["n_significant"]
    <= d_common["n_valid_stations"]
).all(), "n_significant不能超过n_valid_stations"

assert (
    d_common["n_significant"] >= 0
).all(), "n_significant不能为负"

assert (
    d_common["n_valid_stations"] > 0
).all(), "n_valid_stations必须大于0"

assert not d_common["date"].duplicated().any(), \
    "共同样本中存在重复日期"

print(f"严格共同样本天数: {len(d_common)}")
print(
    f"共同样本日期范围: "
    f"{d_common['date'].min().date()} ~ "
    f"{d_common['date'].max().date()}"
)
print("\n共同样本各冬季天数:")
print(d_common["winter"].value_counts(sort=False))

d_common.to_csv(MATCHED_DATA_OUT, index=False)


# ============================================================
# 第三步：构建冬季虚拟变量
# ============================================================

winter_dummies = pd.get_dummies(
    d_common["winter"],
    prefix="winter",
    drop_first=True,
    dtype=float
)

# 主模型字段名统一为 hdd / wind
base = pd.DataFrame(
    {
        "hdd": d_common["hdd_bradford"].astype(float),
        "wind_evening": d_common[
            "wind_speed_ms_evening"
        ].astype(float),
        "wind_allday": d_common[
            "wind_speed_ms_allday"
        ].astype(float)
    },
    index=d_common.index
)

model_data = pd.concat(
    [
        d_common[
            [
                "date",
                "n_significant",
                "n_valid_stations"
            ]
        ],
        base,
        winter_dummies
    ],
    axis=1
)

winter_predictors = winter_dummies.columns.tolist()


# ============================================================
# 第四步：拟合严格匹配样本上的模型
# ============================================================

print("\n" + "=" * 78)
print("第四步：拟合晚间风速与全天风速模型")
print("=" * 78)

all_results = []

# 共同M1
_, res_m1 = fit_grouped_binomial(
    model_data,
    ["hdd"],
    "M1: HDD only"
)
all_results.append(res_m1)

# 晚间风速M2/M3
_, res_m2_evening = fit_grouped_binomial(
    model_data,
    ["hdd", "wind_evening"],
    "M2: HDD + Evening wind"
)
all_results.append(res_m2_evening)

_, res_m3_evening = fit_grouped_binomial(
    model_data,
    ["hdd", "wind_evening", *winter_predictors],
    "M3: HDD + Evening wind + Winter"
)
all_results.append(res_m3_evening)

# 全天风速M2/M3
_, res_m2_allday = fit_grouped_binomial(
    model_data,
    ["hdd", "wind_allday"],
    "M2: HDD + Daily mean wind"
)
all_results.append(res_m2_allday)

_, res_m3_allday = fit_grouped_binomial(
    model_data,
    ["hdd", "wind_allday", *winter_predictors],
    "M3: HDD + Daily mean wind + Winter"
)
all_results.append(res_m3_allday)

results_all = pd.concat(all_results, ignore_index=True)

print_hdd_table(
    results_all,
    "HDD结果：晚间风速 vs 全天平均风速（严格共同样本）"
)

results_all.to_csv(ALL_VARIABLES_OUT, index=False)


# ============================================================
# 第五步：保存HDD和风速摘要
# ============================================================

hdd_summary = results_all[
    results_all["variable"] == "hdd"
].copy()

hdd_summary.to_csv(HDD_SUMMARY_OUT, index=False)

wind_summary = results_all[
    results_all["variable"].isin(
        ["wind_evening", "wind_allday"]
    )
].copy()

wind_summary.to_csv(WIND_SUMMARY_OUT, index=False)

print("\n风速变量结果:")
if not wind_summary.empty:
    print(
        wind_summary[
            [
                "model",
                "variable",
                "beta",
                "OR",
                "CI95_low",
                "CI95_high",
                "p_adjusted",
                "dispersion",
                "n_days"
            ]
        ].round(4).to_string(index=False)
    )


# ============================================================
# 第六步：标准化风速敏感性（便于比较效应强度）
# ============================================================

model_data["wind_evening_z"] = (
    model_data["wind_evening"]
    - model_data["wind_evening"].mean()
) / model_data["wind_evening"].std(ddof=1)

model_data["wind_allday_z"] = (
    model_data["wind_allday"]
    - model_data["wind_allday"].mean()
) / model_data["wind_allday"].std(ddof=1)

_, res_m3_evening_z = fit_grouped_binomial(
    model_data,
    ["hdd", "wind_evening_z", *winter_predictors],
    "M3 standardized: HDD + Evening wind(z) + Winter"
)

_, res_m3_allday_z = fit_grouped_binomial(
    model_data,
    ["hdd", "wind_allday_z", *winter_predictors],
    "M3 standardized: HDD + Daily wind(z) + Winter"
)

standardized_results = pd.concat(
    [res_m3_evening_z, res_m3_allday_z],
    ignore_index=True
)

standardized_results.to_csv(
    OUTPUT_DIR / "sensitivity_wind_window_standardized.csv",
    index=False
)

print("\n标准化风速结果（每增加1个标准差）:")
print(
    standardized_results[
        standardized_results["variable"].isin(
            ["wind_evening_z", "wind_allday_z"]
        )
    ][
        [
            "model",
            "variable",
            "OR",
            "CI95_low",
            "CI95_high",
            "p_adjusted",
            "dispersion"
        ]
    ].round(4).to_string(index=False)
)


# ============================================================
# 第七步：相关性与VIF诊断
# ============================================================

pearson_evening_allday = d_common[
    "wind_speed_ms_evening"
].corr(
    d_common["wind_speed_ms_allday"],
    method="pearson"
)

spearman_evening_allday = d_common[
    "wind_speed_ms_evening"
].corr(
    d_common["wind_speed_ms_allday"],
    method="spearman"
)

pearson_hdd_evening = d_common[
    "hdd_bradford"
].corr(
    d_common["wind_speed_ms_evening"],
    method="pearson"
)

spearman_hdd_evening = d_common[
    "hdd_bradford"
].corr(
    d_common["wind_speed_ms_evening"],
    method="spearman"
)

pearson_hdd_allday = d_common[
    "hdd_bradford"
].corr(
    d_common["wind_speed_ms_allday"],
    method="pearson"
)

spearman_hdd_allday = d_common[
    "hdd_bradford"
].corr(
    d_common["wind_speed_ms_allday"],
    method="spearman"
)

# 分别计算两套M3自变量的VIF
vif_evening_data = pd.concat(
    [
        pd.DataFrame(
            {
                "hdd": d_common["hdd_bradford"],
                "wind_evening": d_common[
                    "wind_speed_ms_evening"
                ]
            },
            index=d_common.index
        ),
        winter_dummies
    ],
    axis=1
)

vif_allday_data = pd.concat(
    [
        pd.DataFrame(
            {
                "hdd": d_common["hdd_bradford"],
                "wind_allday": d_common[
                    "wind_speed_ms_allday"
                ]
            },
            index=d_common.index
        ),
        winter_dummies
    ],
    axis=1
)

vif_evening = calculate_vif(
    vif_evening_data,
    vif_evening_data.columns.tolist()
)
vif_evening["model"] = "M3 Evening wind"

vif_allday = calculate_vif(
    vif_allday_data,
    vif_allday_data.columns.tolist()
)
vif_allday["model"] = "M3 Daily mean wind"

vif_all = pd.concat(
    [vif_evening, vif_allday],
    ignore_index=True
)

diagnostic_rows = [
    {
        "diagnostic": "n_frozen_main_days",
        "value": len(d)
    },
    {
        "diagnostic": "n_matched_days",
        "value": len(d_common)
    },
    {
        "diagnostic": "n_excluded_for_allday_wind",
        "value": int(missing_allday.sum())
    },
    {
        "diagnostic": "matched_start_date",
        "value": str(d_common["date"].min().date())
    },
    {
        "diagnostic": "matched_end_date",
        "value": str(d_common["date"].max().date())
    },
    {
        "diagnostic": "pearson_evening_vs_allday_wind",
        "value": pearson_evening_allday
    },
    {
        "diagnostic": "spearman_evening_vs_allday_wind",
        "value": spearman_evening_allday
    },
    {
        "diagnostic": "pearson_HDD_vs_evening_wind",
        "value": pearson_hdd_evening
    },
    {
        "diagnostic": "spearman_HDD_vs_evening_wind",
        "value": spearman_hdd_evening
    },
    {
        "diagnostic": "pearson_HDD_vs_allday_wind",
        "value": pearson_hdd_allday
    },
    {
        "diagnostic": "spearman_HDD_vs_allday_wind",
        "value": spearman_hdd_allday
    }
]

diagnostics = pd.DataFrame(diagnostic_rows)

# 将VIF以诊断行形式附加
vif_diagnostics = vif_all.assign(
    diagnostic=lambda x: (
        "VIF_"
        + x["model"].str.replace(" ", "_", regex=False)
        + "_"
        + x["variable"]
    ),
    value=lambda x: x["VIF"]
)[["diagnostic", "value"]]

diagnostics = pd.concat(
    [diagnostics, vif_diagnostics],
    ignore_index=True
)

diagnostics.to_csv(DIAGNOSTICS_OUT, index=False)

print("\n" + "=" * 78)
print("相关性诊断")
print("=" * 78)
print(
    f"晚间 vs 全天风速 Pearson r: "
    f"{pearson_evening_allday:.4f}"
)
print(
    f"晚间 vs 全天风速 Spearman rho: "
    f"{spearman_evening_allday:.4f}"
)
print(
    f"HDD vs 晚间风速 Pearson r: "
    f"{pearson_hdd_evening:.4f}"
)
print(
    f"HDD vs 晚间风速 Spearman rho: "
    f"{spearman_hdd_evening:.4f}"
)
print(
    f"HDD vs 全天风速 Pearson r: "
    f"{pearson_hdd_allday:.4f}"
)
print(
    f"HDD vs 全天风速 Spearman rho: "
    f"{spearman_hdd_allday:.4f}"
)

print("\nVIF:")
print(vif_all.round(4).to_string(index=False))


# ============================================================
# 完成
# ============================================================

print("\n" + "=" * 78)
print("全部完成")
print("=" * 78)

print(f"共同样本数据: {MATCHED_DATA_OUT}")
print(f"全部变量结果: {ALL_VARIABLES_OUT}")
print(f"HDD摘要: {HDD_SUMMARY_OUT}")
print(f"风速摘要: {WIND_SUMMARY_OUT}")
print(f"诊断结果: {DIAGNOSTICS_OUT}")
