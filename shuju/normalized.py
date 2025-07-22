import pandas as pd
import numpy as np

# =========================
# STEP 1: 标准化 alpha 因子（含 Winsorization）
# =========================

df = pd.read_csv("alpha001-101.csv")

# 提取因子列名
factor_cols = [col for col in df.columns if col not in ['DATE', 'STOCK_ID']]

# Winsorize 极端值处理
def winsorize(group, limit=5):
    mean = group.mean()
    std = group.std()
    upper = mean + limit * std
    lower = mean - limit * std
    return group.clip(lower=lower, upper=upper)

df[factor_cols] = df.groupby('DATE')[factor_cols].transform(lambda x: winsorize(x, limit=5))

# 标准化
def normalize_factors(group):
    return ((group - group.mean()) / group.std()).fillna(0) * 1e-3

df[factor_cols] = df.groupby('DATE')[factor_cols].transform(normalize_factors)

# =========================
# STEP 2: 读取并处理 train.csv 行情数据
# =========================

train_df = pd.read_csv("train.csv")

# 重命名列名为英文
train_df.columns = ['STOCK_ID', 'DATE', 'open', 'close', 'high', 'low',
                    'volume', 'amount', 'amplitude', 'change', 'turnover', 'pct_chg']

# 排序并计算未来1日收益率
train_df.sort_values(by=['STOCK_ID', 'DATE'], inplace=True)
train_df['return'] = train_df.groupby('STOCK_ID')['close'].pct_change().shift(-1)

# =========================
# STEP 3: 合并 alpha 因子 和 收益率
# =========================

merged = pd.merge(df, train_df[['STOCK_ID', 'DATE', 'return']], on=['STOCK_ID', 'DATE'], how='inner')
merged.dropna(subset=['return'], inplace=True)

# =========================
# STEP 4: 因子 IC 筛选（Spearman）
# =========================

ic_dict = {}
for factor in factor_cols:
    daily_ic = merged.groupby('DATE').apply(
        lambda x: x[[factor, 'return']].corr(method='spearman').iloc[0, 1]
    )
    ic = daily_ic.mean()
    if ic < 0:
        merged[factor] = -merged[factor]
    ic_dict[factor] = ic

# =========================
# STEP 5: 因子收益率与夏普比率
# =========================

# 每日因子收益率 = sum(return * factor)
factor_returns = pd.DataFrame()
for factor in factor_cols:
    factor_returns[factor] = merged.groupby('DATE').apply(lambda x: (x['return'] * x[factor]).sum())

# 夏普比率：200日滚动
window = 200
sharpe_ratio = factor_returns.rolling(window).mean() / factor_returns.rolling(window).std()
sharpe_ratio *= np.sqrt(200)

# 权重：仅保留 Sharpe > 0 的因子，weight = sharpe^2
factor_weights = sharpe_ratio.applymap(lambda x: x**2 if x > 0 else 0)

# 取最新的有效权重
latest_weights = factor_weights.dropna().iloc[-1]

# =========================
# STEP 6: 持仓计算
# =========================

def compute_position(row):
    return sum(row[factor] * latest_weights.get(factor, 0) for factor in factor_cols)

df['position'] = df.apply(compute_position, axis=1)

# =========================
# STEP 7: 保存最终结果
# =========================

df.to_csv("alpha001-101_with_position.csv", index=False)
