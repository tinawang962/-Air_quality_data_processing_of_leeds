# -*- coding: utf-8 -*-
"""
交互模型: Y ~ HDD + Wind + Winter + HDD×Winter
================================================
目的
----
正式检验HDD对晚间峰值概率的斜率, 在三个研究冬季之间是否存在统计意义上
的显著差异(interannual variability), 而不是仅凭分冬季独立拟合的点估计
做主观判断。

关键点
------
- 全部521天在同一个模型里联合拟合, 共享同一个过度离散估计,
  这是跟"分冬季独立拟合"最本质的区别(那种做法每个冬季各自估计
  dispersion, 样本量小、估计不稳定, 且没有正式的假设检验)。
- 交互项 hdd_x_winter2 / hdd_x_winter3 分别代表Winter2/Winter3相对
  Winter1的HDD斜率差异。
- 联合Wald检验回答的问题: 这两个交互项是否同时为0(即三个冬季
  斜率是否相同)。statsmodels自带的wald_test默认不做过度离散调整,
  必须手动用调整后的协方差矩阵重新计算, 否则p值会虚高。

材料需求: day_level_complete_case.csv (已冻结的521天主数据集)
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import chi2
from numpy.linalg import inv

warnings.filterwarnings("ignore")

# ============================================================
# 配置区
# ============================================================
DATA_DIR = Path(r'C:\Users\17793\Desktop\毕业设计材料')
OUTPUT_DIR = DATA_DIR

COMPLETE_CASE_PATH = DATA_DIR / 'day_level_complete_case.csv'
RESULTS_OUT = OUTPUT_DIR / 'interaction_model_results.txt'
COEF_TABLE_OUT = OUTPUT_DIR / 'interaction_model_coefficients.csv'

WINTERS = ['Winter1', 'Winter2', 'Winter3']

# ============================================================
# 第一步: 读入数据
# ============================================================
d = pd.read_csv(COMPLETE_CASE_PATH)

required_cols = ['hdd_bradford', 'wind_speed_ms_evening', 'winter',
                  'n_valid_stations', 'n_significant']
missing = set(required_cols) - set(d.columns)
if missing:
    raise ValueError(f"缺少必要列: {missing}")

d['winter'] = pd.Categorical(d['winter'], categories=WINTERS, ordered=True)

print(f"样本量: n={len(d)}")
print(d['winter'].value_counts(sort=False))

# ============================================================
# 第二步: 构造设计矩阵, 含HDD×Winter交互项
# ============================================================
y = np.column_stack([
    d['n_significant'].astype(float),
    (d['n_valid_stations'] - d['n_significant']).astype(float)
])

winter_dummies = pd.get_dummies(d['winter'], prefix='winter', drop_first=True).astype(float)

X = pd.DataFrame({
    'hdd': d['hdd_bradford'].astype(float),
    'wind_evening': d['wind_speed_ms_evening'].astype(float),
})
X = pd.concat([X, winter_dummies], axis=1)

# 交互项: HDD x Winter2, HDD x Winter3 (Winter1作为参照组, 交互项天然为0)
X['hdd_x_winter2'] = d['hdd_bradford'] * winter_dummies['winter_Winter2']
X['hdd_x_winter3'] = d['hdd_bradford'] * winter_dummies['winter_Winter3']

X = sm.add_constant(X)

# ============================================================
# 第三步: 拟合模型, 做过度离散调整
# ============================================================
model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
dispersion = model.pearson_chi2 / model.df_resid
se_adj = model.bse * np.sqrt(dispersion)
z_adj = model.params / se_adj
p_adj = 2 * (1 - stats.norm.cdf(np.abs(z_adj)))

print(f"\n过度离散系数(单一联合估计, 全部变量共享): {dispersion:.4f}")
print("\n" + "=" * 95)
print(f"交互模型: Y ~ HDD + Wind + Winter + HDD×Winter (n={len(d)})")
print("=" * 95)
print(f"{'变量':<20}{'系数':>10}{'调整后SE':>12}{'调整后p值':>12}{'OR':>10}")

coef_rows = []
for var in X.columns:
    if var == 'const':
        continue
    coef = model.params[var]
    se = se_adj[var]
    p = p_adj[X.columns.get_loc(var)]
    print(f"{var:<20}{coef:>10.4f}{se:>12.4f}{p:>12.4f}{np.exp(coef):>10.4f}")
    coef_rows.append({
        'variable': var, 'beta': coef, 'SE_adjusted': se,
        'OR': np.exp(coef), 'CI95_low': np.exp(coef - 1.96*se),
        'CI95_high': np.exp(coef + 1.96*se), 'p_adjusted': p,
        'dispersion': dispersion, 'n_days': len(d),
    })

pd.DataFrame(coef_rows).to_csv(COEF_TABLE_OUT, index=False)

# ============================================================
# 第四步: 联合Wald检验 —— H0: hdd_x_winter2 = hdd_x_winter3 = 0
# 必须手动用过度离散调整后的协方差矩阵, 不能直接用statsmodels默认的wald_test
# (默认版本假设scale=1, 会让p值虚高, 跟本分析全文的过度离散调整原则不一致)
# ============================================================
print("\n" + "=" * 95)
print("联合Wald检验: H0: hdd_x_winter2 = hdd_x_winter3 = 0 (三个冬季HDD斜率是否相同)")
print("=" * 95)

# 未调整版本(仅作对照展示, 不作为结论依据)
wald_naive = model.wald_test('hdd_x_winter2 = 0, hdd_x_winter3 = 0', scalar=True)
print(f"[对照, 未过度离散调整]: {wald_naive}")

# 正确版本: 用dispersion缩放协方差矩阵后重新做Wald检验
cov_adj = model.cov_params() * dispersion
R = np.zeros((2, len(X.columns)))
R[0, X.columns.get_loc('hdd_x_winter2')] = 1
R[1, X.columns.get_loc('hdd_x_winter3')] = 1
beta = model.params.values
Rb = R @ beta
cov_Rb = R @ cov_adj.values @ R.T
wald_stat = Rb @ inv(cov_Rb) @ Rb
p_wald_adj = 1 - chi2.cdf(wald_stat, df=2)

print(f"\n[正确, 过度离散调整后]: Wald chi2 = {wald_stat:.4f}, df=2, p = {p_wald_adj:.4f}")
print("(这是应当在论文里引用的检验结果)")

# ============================================================
# 第五步: 从交互模型系数还原各冬季隐含的HDD斜率(便于对照分冬季独立拟合)
# ============================================================
print("\n各冬季HDD斜率(从交互模型系数还原, Winter1为参照基准):")
hdd_w1 = model.params['hdd']
hdd_w2 = model.params['hdd'] + model.params['hdd_x_winter2']
hdd_w3 = model.params['hdd'] + model.params['hdd_x_winter3']
print(f"Winter1: {hdd_w1:.4f}")
print(f"Winter2: {hdd_w2:.4f}  (= {model.params['hdd']:.4f} + {model.params['hdd_x_winter2']:.4f})")
print(f"Winter3: {hdd_w3:.4f}  (= {model.params['hdd']:.4f} + {model.params['hdd_x_winter3']:.4f})")

# ============================================================
# 保存完整结果
# ============================================================
with open(RESULTS_OUT, 'w', encoding='utf-8') as f:
    f.write(str(model.summary()))
    f.write(f"\n\n过度离散系数: {dispersion:.4f}\n")
    f.write(f"\n联合Wald检验(过度离散调整后): chi2={wald_stat:.4f}, df=2, p={p_wald_adj:.4f}\n")
    f.write(f"\n各冬季隐含HDD斜率:\n")
    f.write(f"Winter1: {hdd_w1:.4f}\nWinter2: {hdd_w2:.4f}\nWinter3: {hdd_w3:.4f}\n")

print(f"\n完整结果已保存: {RESULTS_OUT}")
print(f"系数表已保存: {COEF_TABLE_OUT}")
