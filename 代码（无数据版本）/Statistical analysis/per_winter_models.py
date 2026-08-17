# -*- coding: utf-8 -*-
"""
分冬季模型: Y ~ HDD 和 Y ~ HDD + Evening Wind
================================================
目的
----
不追求每个冬季单独显著, 而是判断HDD效应的方向在三个冬季是否一致,
尤其确认完整补回2025年1-3月数据后, Winter3的表现是否依然与Winter1/2
方向相反或量级明显更弱。

固定不变
--------
- HDD来源: Bradford
- 风速: Bramham 18:00-23:00 晚间平均
- 响应变量: SignificantEveningPeak (阈值14.87 µg/m³, 不改动)
- 事件日排除规则: 主分析版本
- 使用日层面分组二项GLM, Pearson chi2/df 调整过度离散

材料需求: day_level_complete_case.csv (已冻结的521天主数据集)
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

# ============================================================
# 配置区
# ============================================================
DATA_DIR = Path(r'C:\Users\17793\Desktop\毕业设计材料')
OUTPUT_DIR = DATA_DIR

COMPLETE_CASE_PATH = DATA_DIR / 'day_level_complete_case.csv'
OUTPUT_PATH = OUTPUT_DIR / 'per_winter_models.csv'

WINTERS = ['Winter1', 'Winter2', 'Winter3']


# ============================================================
# 工具函数
# ============================================================
def two_sided_normal_p(z):
    return 2 * stats.norm.sf(abs(z))


def fit_grouped_binomial(data, predictors, label):
    """拟合日层面分组二项GLM, 返回过度离散调整后的完整结果"""
    y = np.column_stack([
        data['n_significant'].astype(float),
        (data['n_valid_stations'] - data['n_significant']).astype(float)
    ])
    X = data[predictors].astype(float).copy()
    X = sm.add_constant(X, has_constant='add')

    model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    dispersion = model.pearson_chi2 / model.df_resid

    se_adj = model.bse * np.sqrt(dispersion)
    z_adj = model.params / se_adj
    p_adj = pd.Series([two_sided_normal_p(z) for z in z_adj], index=model.params.index)

    rows = []
    for var in model.params.index:
        beta = model.params[var]
        se = se_adj[var]
        rows.append({
            'winter': label.split(' | ')[0],
            'model': label.split(' | ')[1],
            'variable': var,
            'beta': beta,
            'SE_adjusted': se,
            'OR': np.exp(beta),
            'CI95_low': np.exp(beta - 1.96 * se),
            'CI95_high': np.exp(beta + 1.96 * se),
            'p_adjusted': p_adj[var],
            'dispersion': dispersion,
            'n_days': len(data),
        })
    return pd.DataFrame(rows)


# ============================================================
# 第一步: 读入冻结数据, 基本检查
# ============================================================
d = pd.read_csv(COMPLETE_CASE_PATH)
d['date'] = pd.to_datetime(d['date'], errors='coerce')

required_cols = ['date', 'hdd_bradford', 'wind_speed_ms_evening', 'winter',
                  'n_valid_stations', 'n_significant']
missing = set(required_cols) - set(d.columns)
if missing:
    raise ValueError(f"缺少必要列: {missing}")

d['winter'] = pd.Categorical(d['winter'], categories=WINTERS, ordered=True)

print(f"总样本数: {len(d)}")
print(d['winter'].value_counts(sort=False))

# ============================================================
# 第二步: 逐冬季拟合 M_HDD 和 M_HDD+Wind
# ============================================================
all_results = []

for w in WINTERS:
    sub = d[d['winter'] == w].copy()
    sub = sub.dropna(subset=['hdd_bradford', 'wind_speed_ms_evening'])
    print(f"\n{w}: n={len(sub)}")

    if len(sub) < 10:
        print(f"  样本量过小, 跳过")
        continue

    res_hdd = fit_grouped_binomial(sub, ['hdd_bradford'], f"{w} | Y~HDD")
    res_hdd = res_hdd.rename(columns={'variable': 'variable'})
    res_hdd.loc[res_hdd['variable'] == 'hdd_bradford', 'variable'] = 'hdd'
    all_results.append(res_hdd)

    res_hdd_wind = fit_grouped_binomial(
        sub, ['hdd_bradford', 'wind_speed_ms_evening'], f"{w} | Y~HDD+Wind"
    )
    res_hdd_wind.loc[res_hdd_wind['variable'] == 'hdd_bradford', 'variable'] = 'hdd'
    res_hdd_wind.loc[res_hdd_wind['variable'] == 'wind_speed_ms_evening', 'variable'] = 'wind_evening'
    all_results.append(res_hdd_wind)

results = pd.concat(all_results, ignore_index=True)

# ============================================================
# 第三步: 打印HDD结果汇总表(核心输出)
# ============================================================
hdd_results = results[results['variable'] == 'hdd'].copy()

print("\n" + "=" * 100)
print("分冬季 HDD 效应汇总 (方向一致性检查)")
print("=" * 100)
print(f"{'Winter':<10}{'Model':<15}{'HDD beta':>10}{'OR':>10}{'95% CI':>22}{'adj. p':>10}{'n':>6}")
for _, row in hdd_results.iterrows():
    ci = f"{row['CI95_low']:.4f}-{row['CI95_high']:.4f}"
    print(f"{row['winter']:<10}{row['model']:<15}{row['beta']:>10.4f}{row['OR']:>10.4f}"
          f"{ci:>22}{row['p_adjusted']:>10.4f}{int(row['n_days']):>6}")

results.to_csv(OUTPUT_PATH, index=False)
print(f"\n完整结果已保存: {OUTPUT_PATH}")
