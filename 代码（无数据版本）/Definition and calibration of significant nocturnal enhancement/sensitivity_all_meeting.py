# -*- coding: utf-8 -*-
"""
完整敏感性分析脚本: 阈值定义 / 阈值基线 / 气象站选择 / 风速窗口
================================================================
每一类检验, 都同时报告"仅HDD"和"HDD+Wind+Winter"两个版本, 目的是
一致地展示核心发现: HDD单独看时几乎全部显著, 加入风速和冬季控制后
几乎全部不再显著——这个模式在四类不同的敏感性检验里都稳健存在。

固定不变
--------
- 主分析样本: day_level_complete_case.csv 对应的521天(排除事件日后)
- 响应变量阈值: 90th percentile = 14.87 µg/m3(除非本身就是被检验对象)
- 过度离散调整: Pearson chi2/df, 全部推断均已调整

材料需求: daily_peak_stats_remerged.csv (已冻结的最终station-day数据集)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r'C:\Users\17793\Desktop\毕业设计材料'
OUTPUT_DIR = DATA_DIR

# ============================================================
# 通用拟合函数
# ============================================================
def fit(dd, hdd_col='hdd_bradford', wind_col='wind_speed_ms_evening', include_wind_winter=True):
    day_agg = dd.groupby('date').agg(
        n_valid_stations=('significant_evening_peak', 'size'),
        n_significant=('significant_evening_peak', 'sum'),
        hdd=(hdd_col, 'first'), wind=(wind_col, 'first'), winter=('winter', 'first'),
    ).reset_index()
    d3 = day_agg.dropna(subset=['hdd', 'wind', 'winter']).copy()
    y = np.column_stack([d3['n_significant'], d3['n_valid_stations'] - d3['n_significant']])
    X = pd.DataFrame({'hdd': d3['hdd']})
    if include_wind_winter:
        X['wind'] = d3['wind']
        wd = pd.get_dummies(d3['winter'], prefix='winter', drop_first=True).astype(float)
        X = pd.concat([X, wd], axis=1)
    X = sm.add_constant(X)
    m = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    disp = m.pearson_chi2 / m.df_resid
    se = m.bse['hdd'] * np.sqrt(disp)
    p = 2 * (1 - stats.norm.cdf(abs(m.params['hdd'] / se)))
    ci_lo, ci_hi = np.exp(m.params['hdd'] - 1.96 * se), np.exp(m.params['hdd'] + 1.96 * se)
    return {'n': len(d3), 'coef': round(m.params['hdd'], 4), 'OR': round(np.exp(m.params['hdd']), 4),
            'CI_lo': round(ci_lo, 4), 'CI_hi': round(ci_hi, 4), 'p': round(p, 4), 'dispersion': round(disp, 4)}

# ============================================================
# 读入数据
# ============================================================
daily_raw = pd.read_csv(f'{DATA_DIR}\\daily_peak_stats_remerged.csv')
daily_raw['date'] = pd.to_datetime(daily_raw['date'])
daily_raw = daily_raw[~daily_raw['event_day']].copy()

all_results = []

# ============================================================
# 检验一: 阈值定义敏感性 (90th/95th/固定+4/+5/+6)
# ============================================================
thresholds = {'90th (14.87, main)': 14.87, '95th (19.48)': 19.48,
              'Fixed +4': 4, 'Fixed +5': 5, 'Fixed +6': 6}
for label, thresh in thresholds.items():
    d2 = daily_raw.copy()
    d2['significant_evening_peak'] = ((d2['evening_peak_flag'] == 1) & (d2['evening_peak_enhancement'] > thresh)).astype(int)
    r_hdd = fit(d2, include_wind_winter=False)
    r_full = fit(d2, include_wind_winter=True)
    all_results.append({'category': '1. Threshold definition', 'variant': label, 'model': 'HDD only', **r_hdd})
    all_results.append({'category': '1. Threshold definition', 'variant': label, 'model': 'HDD+Wind+Winter', **r_full})

# ============================================================
# 检验二: 气象站敏感性 (Bradford vs Bramham)
# ============================================================
for label, hdd_col in [('Bradford (main)', 'hdd_bradford'), ('Bramham', 'hdd_bramham')]:
    r_hdd = fit(daily_raw, hdd_col=hdd_col, include_wind_winter=False)
    r_full = fit(daily_raw, hdd_col=hdd_col, include_wind_winter=True)
    all_results.append({'category': '2. Met station', 'variant': label, 'model': 'HDD only', **r_hdd})
    all_results.append({'category': '2. Met station', 'variant': label, 'model': 'HDD+Wind+Winter', **r_full})

# ============================================================
# 检验三: 阈值基线敏感性 (5-9月 vs 4-9月非供暖季基线)
# 需要预先算好的两个基线阈值(见threshold_baseline_sensitivity.py)
# ============================================================
baseline_thresholds = {'5-9\u6708 (14.87, main)': 14.87, '4-9\u6708 (15.06)': 15.0561}
for label, thresh in baseline_thresholds.items():
    d2 = daily_raw.copy()
    d2['significant_evening_peak'] = ((d2['evening_peak_flag'] == 1) & (d2['evening_peak_enhancement'] > thresh)).astype(int)
    r_hdd = fit(d2, include_wind_winter=False)
    r_full = fit(d2, include_wind_winter=True)
    all_results.append({'category': '3. Baseline period', 'variant': label, 'model': 'HDD only', **r_hdd})
    all_results.append({'category': '3. Baseline period', 'variant': label, 'model': 'HDD+Wind+Winter', **r_full})

df_all = pd.DataFrame(all_results)
print(df_all.to_string(index=False))
df_all.to_csv(f'{OUTPUT_DIR}\\sensitivity_all_with_without_wind.csv', index=False)
print(f"\n已保存: sensitivity_all_with_without_wind.csv")

# ============================================================
# 检验四: 风速窗口敏感性 (晚间 vs 全天) —— 见独立脚本 sensitivity_wind_window.py
# (因为需要额外加载逐小时风速数据重新计算全天平均, 逻辑较独立, 保持单独脚本)
# ============================================================
print("\n注: 检验四(风速窗口敏感性)见配套脚本 sensitivity_wind_window.py, "
      "结果见 sensitivity_wind_window_all_variables.csv / sensitivity_wind_window_hdd_summary.csv")
