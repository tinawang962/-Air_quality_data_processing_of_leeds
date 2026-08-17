# -*- coding: utf-8 -*-
"""
阈值校准基线敏感性检验: 5-9月 vs 4-9月
================================================
目的
----
检验非供暖季阈值校准基线是否包含4月, 对主结论(HDD效应主要被风速解释)
是否有实质影响。呼应Methods 2.5节末尾预先声明的敏感性检验计划。

固定不变
--------
- HDD来源: Bradford; 风速: Bramham晚间均值; 事件日排除规则: 主分析版本
- 每个基线下, 同时报告"仅HDD"和"HDD+Wind+Winter"两个模型, 
  以便同时检验基线选择的稳健性, 以及"风速解释HDD效应"这一核心发现
  在不同基线下是否依然成立

材料需求
--------
1. nonheating_corrected/SL0xx_nonheating_corrected.csv (5-9月矫正数据)
2. april_corrected/SL0xx_april_corrected.csv (4月矫正数据)
3. daily_peak_stats_remerged.csv (已冻结的最终station-day数据集,
   含evening_peak_flag, evening_peak_enhancement, hdd_bradford,
   wind_speed_ms_evening, winter, event_day)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置区
# ============================================================
DATA_DIR = r'C:\Users\17793\Desktop\毕业设计材料'
STATIONS = ['SL001', 'SL005', 'SL006', 'SL007', 'SL017', 'SL018']
MIN_HOURS_PER_DAY = 18

# ============================================================
# 第一步: 从矫正后小时数据算station-day层面的peak enhancement
# ============================================================
def load_and_compute_daily(files, min_hours=MIN_HOURS_PER_DAY):
    all_daily = []
    for station, path in files.items():
        df = pd.read_csv(path)
        df['hour'] = pd.to_datetime(df['hour'])
        df['date'] = df['hour'].dt.normalize()
        daily = df.groupby('date').agg(
            n_hours=('pm25_corrected', 'count'),
            pm_max=('pm25_corrected', 'max'),
            pm_median=('pm25_corrected', 'median'),
        ).reset_index()
        daily = daily[daily['n_hours'] >= min_hours].copy()
        daily['peak_enhancement'] = daily['pm_max'] - daily['pm_median']
        daily['station'] = station
        all_daily.append(daily)
    return pd.concat(all_daily, ignore_index=True)

files_may_sep = {s: f'{DATA_DIR}\\{s}_nonheating_corrected.csv' for s in STATIONS}
daily_may_sep = load_and_compute_daily(files_may_sep)
p90_may_sep = daily_may_sep['peak_enhancement'].quantile(0.90)
print(f"5-9月基线: n={len(daily_may_sep)}, 90分位={p90_may_sep:.4f}")

files_april = {s: f'{DATA_DIR}\\{s}_april_corrected.csv' for s in STATIONS}
daily_april = load_and_compute_daily(files_april)
daily_apr_sep = pd.concat([daily_april, daily_may_sep], ignore_index=True)
p90_apr_sep = daily_apr_sep['peak_enhancement'].quantile(0.90)
print(f"4-9月基线: n={len(daily_apr_sep)}, 90分位={p90_apr_sep:.4f}")

# ============================================================
# 第二步: 拟合函数(仅HDD / HDD+Wind+Winter两种)
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
    ci_lo, ci_hi = np.exp(m.params['hdd'] - 1.96*se), np.exp(m.params['hdd'] + 1.96*se)
    return {'n': len(d3), 'coef': m.params['hdd'], 'OR': np.exp(m.params['hdd']),
            'CI_lo': ci_lo, 'CI_hi': ci_hi, 'p': p, 'dispersion': disp}

# ============================================================
# 第三步: 两种基线 x 两种模型, 共四组结果
# ============================================================
daily_raw = pd.read_csv(f'{DATA_DIR}\\daily_peak_stats_remerged.csv')
daily_raw['date'] = pd.to_datetime(daily_raw['date'])
daily_raw = daily_raw[~daily_raw['event_day']].copy()

results = []
for label, thresh in [('5-9\u6708 (\u4e3b\u5206\u6790)', p90_may_sep), ('4-9\u6708 (\u654f\u611f\u6027\u68c0\u9a8c)', p90_apr_sep)]:
    d2 = daily_raw.copy()
    d2['significant_evening_peak'] = ((d2['evening_peak_flag'] == 1) & (d2['evening_peak_enhancement'] > thresh)).astype(int)
    r_hdd = fit(d2, include_wind_winter=False)
    r_full = fit(d2, include_wind_winter=True)
    results.append({'baseline': label, 'model': 'HDD only', **r_hdd})
    results.append({'baseline': label, 'model': 'HDD+Wind+Winter', **r_full})

df = pd.DataFrame(results)
print(df.to_string(index=False))
df.to_csv(f'{DATA_DIR}\\threshold_baseline_sensitivity_final.csv', index=False)
print("\n已保存: threshold_baseline_sensitivity_final.csv")
