# -*- coding: utf-8 -*-
"""
Figure 3: 非供暖季PM2.5峰值抬升分布图(阈值校准)
================================================
对应Results 3.3节。展示非供暖季(5-9月)六站合并的peak enhancement分布,
标出90分位阈值(14.87 µg/m3)的具体位置。

材料需求: SL0xx_nonheating_corrected.csv (六个站点的非供暖季矫正数据)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MIN_HOURS_PER_DAY = 18
STATIONS = ['SL001', 'SL005', 'SL006', 'SL007', 'SL017', 'SL018']
DATA_DIR = r'C:\Users\17793\Desktop\毕业设计材料'
OUTPUT_PATH = f'{DATA_DIR}\\figure3_nonheating_threshold_distribution.png'

# ============================================================
# 第一步: 从六站非供暖季矫正数据算出station-day层面的peak enhancement
# ============================================================
all_daily = []
for station in STATIONS:
    path = f'{DATA_DIR}\\{station}_nonheating_corrected.csv'
    df = pd.read_csv(path)
    df['hour'] = pd.to_datetime(df['hour'])
    df['date'] = df['hour'].dt.normalize()
    daily = df.groupby('date').agg(
        n_hours=('pm25_corrected', 'count'),
        pm_max=('pm25_corrected', 'max'),
        pm_median=('pm25_corrected', 'median'),
    ).reset_index()
    daily = daily[daily['n_hours'] >= MIN_HOURS_PER_DAY].copy()
    daily['peak_enhancement'] = daily['pm_max'] - daily['pm_median']
    all_daily.append(daily)

daily_nonheating = pd.concat(all_daily, ignore_index=True)
n = len(daily_nonheating)
p90 = daily_nonheating['peak_enhancement'].quantile(0.90)
print(f"非供暖季station-day数: n={n}")
print(f"90分位阈值: {p90:.4f} \u00b5g/m3")

# ============================================================
# 第二步: 画图
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(daily_nonheating['peak_enhancement'], bins=60, color='steelblue',
        edgecolor='white', alpha=0.85)
ax.axvline(p90, color='firebrick', linestyle='--', lw=2,
           label=f'90th percentile = {p90:.2f} \u00b5g/m\u00b3')
ax.set_xlabel('Peak enhancement (daily max \u2212 daily median), \u00b5g/m\u00b3')
ax.set_ylabel('Count of station-days')
ax.set_title(f'Non-heating Season Peak Enhancement Distribution\n'
             f'(May\u2013September 2023 and 2024, n={n} station-days)')
ax.legend()
ax.set_xlim(0, daily_nonheating['peak_enhancement'].quantile(0.995))

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
print(f"\n图已保存: {OUTPUT_PATH}")
