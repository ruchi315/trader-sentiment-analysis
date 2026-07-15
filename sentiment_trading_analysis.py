"""
Trader Performance vs. Bitcoin Market Sentiment - Full Analysis
==================================================================
Merges Hyperliquid trader execution data with the Bitcoin Fear & Greed
Index and analyzes the relationship between market sentiment and
trading performance (PnL, win rate, position sizing, direction bias).

Inputs (place in the same folder or update paths below):
    - historical_data.csv        (Hyperliquid trade executions)
    - fear_greed_index.csv       (Bitcoin Fear & Greed Index)

Outputs:
    - merged.pkl, daily.pkl      (intermediate data)
    - chart1-5 .png              (visualizations)
    - printed console summary of all key stats
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

pd.set_option('display.max_columns', None)

TRADES_PATH = '/mnt/user-data/uploads/historical_data.csv'
FG_PATH = '/mnt/user-data/uploads/fear_greed_index.csv'

# ---------------------------------------------------------------
# 1. LOAD & MERGE
# ---------------------------------------------------------------
trades = pd.read_csv(TRADES_PATH)
fg = pd.read_csv(FG_PATH)

trades['dt'] = pd.to_datetime(trades['Timestamp IST'], format='%d-%m-%Y %H:%M')
trades['date'] = pd.to_datetime(trades['dt'].dt.date)

fg['date'] = pd.to_datetime(fg['date'])
fg = fg[['date', 'value', 'classification']].rename(
    columns={'value': 'fg_value', 'classification': 'fg_class'})

merged = trades.merge(fg, on='date', how='left')

def simplify(c):
    if c in ['Extreme Fear', 'Fear']:
        return 'Fear'
    elif c in ['Extreme Greed', 'Greed']:
        return 'Greed'
    return 'Neutral'

merged['sentiment_bucket'] = merged['fg_class'].apply(simplify)
print(f"Total trades: {len(trades):,} | Matched to sentiment: {merged['fg_class'].notna().sum():,}")

# ---------------------------------------------------------------
# 2. PERFORMANCE METRICS BY SENTIMENT
# ---------------------------------------------------------------
order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
realized = merged[merged['Closed PnL'] != 0]

grp5 = realized.groupby('fg_class').agg(
    trades=('Closed PnL', 'count'),
    avg_pnl=('Closed PnL', 'mean'),
    win_rate=('Closed PnL', lambda x: (x > 0).mean())
).reindex(order).round(3)
print("\n=== Realized PnL & Win Rate by Sentiment Class ===")
print(grp5)

# ---------------------------------------------------------------
# 3. STATISTICAL TESTS
# ---------------------------------------------------------------
fear_pnl = realized[realized['sentiment_bucket'] == 'Fear']['Closed PnL']
greed_pnl = realized[realized['sentiment_bucket'] == 'Greed']['Closed PnL']

t_fg, p_fg = stats.ttest_ind(fear_pnl, greed_pnl, equal_var=False)
u_fg, pu_fg = stats.mannwhitneyu(fear_pnl, greed_pnl, alternative='two-sided')

win_fear, lose_fear = (fear_pnl > 0).sum(), (fear_pnl < 0).sum()
win_greed, lose_greed = (greed_pnl > 0).sum(), (greed_pnl < 0).sum()
table = np.array([[win_fear, lose_fear], [win_greed, lose_greed]])
chi2, p_chi, _, _ = stats.chi2_contingency(table)

corr_trade = merged[['fg_value', 'Closed PnL']].corr().iloc[0, 1]
daily = merged.groupby('date').agg(
    daily_pnl=('Closed PnL', 'sum'), fg_value=('fg_value', 'first')).reset_index()
corr_daily = daily[['fg_value', 'daily_pnl']].corr().iloc[0, 1]

print("\n=== Statistical Tests: Fear vs Greed ===")
print(f"Welch t-test (avg PnL):        t={t_fg:.3f}, p={p_fg:.4f}")
print(f"Mann-Whitney U (PnL dist.):     U={u_fg:.0f}, p={pu_fg:.4f}")
print(f"Chi-square (win rate):          chi2={chi2:.3f}, p={p_chi:.6f}")
print(f"Correlation fg_value~PnL (trade level): r={corr_trade:.4f}")
print(f"Correlation fg_value~PnL (daily level):  r={corr_daily:.4f}")

# ---------------------------------------------------------------
# 4. TRADING BEHAVIOR (volume, size, direction)
# ---------------------------------------------------------------
behavior = merged.groupby('fg_class').agg(
    total_volume_usd=('Size USD', 'sum'),
    avg_size_usd=('Size USD', 'mean')).reindex(order).round(2)
print("\n=== Trading Volume & Position Size by Sentiment ===")
print(behavior)

open_trades = merged[merged['Direction'].isin(['Open Long', 'Open Short'])]
direction = open_trades.groupby(['fg_class', 'Direction']).size().unstack().reindex(order)
print("\n=== Long vs Short Positions Opened by Sentiment ===")
print(direction)

# ---------------------------------------------------------------
# 5. TRADER-LEVEL HETEROGENEITY
# ---------------------------------------------------------------
acct = realized.groupby(['Account', 'sentiment_bucket']).agg(
    trades=('Closed PnL', 'count'),
    avg_pnl=('Closed PnL', 'mean'),
    win_rate=('Closed PnL', lambda x: (x > 0).mean())).round(2)
print("\n=== Sample: Per-Account Win Rate by Sentiment (first 20 rows) ===")
print(acct.head(20))

# ---------------------------------------------------------------
# 6. VISUALIZATIONS
# ---------------------------------------------------------------
colors = {'Extreme Fear': '#8B0000', 'Fear': '#E67E22', 'Neutral': '#95A5A6',
          'Greed': '#27AE60', 'Extreme Greed': '#145A32'}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].bar(grp5.index, grp5['avg_pnl'], color=[colors[c] for c in order])
axes[0].set_title('Avg Realized PnL by Sentiment'); axes[0].tick_params(axis='x', rotation=30)
axes[1].bar(grp5.index, grp5['win_rate'] * 100, color=[colors[c] for c in order])
axes[1].set_title('Win Rate by Sentiment (%)'); axes[1].tick_params(axis='x', rotation=30)
plt.tight_layout(); plt.savefig('chart1_pnl_winrate.png'); plt.close()

daily_sorted = daily.sort_values('date')
daily_sorted['cum_pnl'] = daily_sorted['daily_pnl'].cumsum()
fig, ax1 = plt.subplots(figsize=(13, 5))
ax1.plot(daily_sorted['date'], daily_sorted['cum_pnl'], color='black', linewidth=1.3)
ax1.set_ylabel('Cumulative Realized PnL (USD)')
ax2 = ax1.twinx()
ax2.fill_between(daily_sorted['date'], daily_sorted['fg_value'], color='steelblue', alpha=0.15)
ax2.set_ylabel('Fear/Greed Index', color='steelblue'); ax2.set_ylim(0, 100)
ax1.set_title('Cumulative PnL vs Fear/Greed Index Over Time')
plt.tight_layout(); plt.savefig('chart3_cumulative_pnl_sentiment.png'); plt.close()

print("\nAll charts saved to current directory.")
