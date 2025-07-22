
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import math

# ========== Data Preparation ==========

def prepare_data(df):
    df = df.sort_values(['DATE', 'STOCK_ID']).copy()
    #df = df[(df["STOCK_ID"] == 600674)|(df["STOCK_ID"] == 601058)]
    df.set_index(['DATE', 'STOCK_ID'], inplace=True)
    return df

# ========== Auxiliary Functions ==========

def ts_sum(df, window=10):
    window = math.ceil(window)	
    return df.groupby('STOCK_ID').rolling(window).sum().droplevel(0)

def sma(df, window=10):
    window = math.ceil(window)	
    return df.groupby('STOCK_ID').rolling(window).mean().droplevel(0)

def stddev(df, window=10):
    window = math.ceil(window)	
    return df.groupby('STOCK_ID').rolling(window).std().droplevel(0)

# def correlation(x, y, window=10):
#     return x.groupby('STOCK_ID').rolling(window).corr(y).droplevel(0)

def correlation(x, y, window=10):
    window = math.ceil(window)	
    df = pd.concat([x, y], axis=1)
    df.columns = ['x', 'y']
    df = df.reset_index().set_index(['STOCK_ID', 'DATE']).sort_index()

    def rolling_corr(group):
        return group['x'].rolling(window).corr(group['y'])

    return df.groupby('STOCK_ID').apply(rolling_corr).reset_index(level=0, drop=True)

# def covariance(x, y, window=10):
#     return x.groupby('STOCK_ID').rolling(window).cov(y).droplevel(0)
def covariance(x, y, window=10):
    window = math.ceil(window)	
    df = pd.concat([x, y], axis=1)
    df.columns = ['x', 'y']
    df = df.reset_index().set_index(['STOCK_ID', 'DATE']).sort_index()

    def rolling_cov(group):
        return group['x'].rolling(window).cov(group['y'])

    return df.groupby('STOCK_ID').apply(rolling_cov).reset_index(level=0, drop=True)

def ts_rank(df, window=10):
    window = math.ceil(window)	
    return df.groupby('STOCK_ID').rolling(window).apply(lambda x: rankdata(x)[-1], raw=True).droplevel(0)

def product(df, window=10):
    window = math.ceil(window)	
    return df.groupby('STOCK_ID').rolling(window).apply(np.prod, raw=True).droplevel(0)

def ts_min(df, window=10):
    window = math.ceil(window)	
    return df.groupby('STOCK_ID').rolling(window).min().droplevel(0)

def ts_max(df, window=10):
    window = math.ceil(window)	
    return df.groupby('STOCK_ID').rolling(window).max().droplevel(0)

def delta(df, period=1):
    return df.groupby('STOCK_ID').diff(period)

def delay(df, period=1):
    return df.groupby('STOCK_ID').shift(period)

def rank(df):
    return df.groupby('DATE').rank(pct=True)

def scale(df, k=1):
    return df.groupby('STOCK_ID').apply(lambda x: x * k / np.abs(x).sum())

def ts_argmax(df, window=10):
    window = math.ceil(window)	
    return df.groupby('STOCK_ID').rolling(window).apply(np.argmax, raw=True).droplevel(0) + 1

def ts_argmin(df, window=10):
    window = math.ceil(window)	
    return df.groupby('STOCK_ID').rolling(window).apply(np.argmin, raw=True).droplevel(0) + 1

def sign(df):
    return np.sign(df)

def log(df):
    return np.log(df.replace(0, np.nan) + 1)

def decay_linear(df, window=10):
    window = math.ceil(window)	
    def weighted_mean(x):
        weights = np.arange(1, window + 1)
        return (x * weights).sum() / weights.sum()

    return df.groupby('STOCK_ID').rolling(window).apply(weighted_mean, raw=True).droplevel(0)

# ========== Alphas Class with alpha001 - alpha033 ==========

class Alphas:
    def __init__(self, df_data):
        self.open = df_data['OPEN']
        self.high = df_data['HIGH']
        self.low = df_data['LOW']
        self.close = df_data['CLOSE']
        self.volume = df_data['VOLUME'] * 100
        self.returns = df_data['PCTCHANGE']
        self.vwap = (df_data['AMOUNT']) / (df_data['VOLUME'] * 1000)
    
    #Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5) 
    def alpha001(self):
        inner = self.close
        inner = inner.copy()
        inner[self.returns < 0] = stddev(self.returns, 20)
        return rank(ts_argmax(inner ** 2, 5)) - 0.5
    
        
    # Alpha#2	 (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    def alpha002(self):
        df = -1 * correlation(rank(delta(log(self.volume), 2)), rank(((self.close - self.open) / self.open)), 6)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)
    
    # Alpha#3	 (-1 * correlation(rank(open), rank(volume), 10))
    def alpha003(self):
        df = -1 * correlation(rank(self.open), rank(self.volume), 10)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)
        
    # Alpha#4	 (-1 * Ts_Rank(rank(low), 9))
    def alpha004(self):
        return -1 * ts_rank(rank(self.low), 9)
        
    # Alpha#5	 (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    def alpha005(self):
        return  (rank((self.open - (sum(self.vwap, 10) / 10))) * (-1 * abs(rank((self.close - self.vwap)))))
        
    # Alpha#6	 (-1 * correlation(open, volume, 10))
    def alpha006(self):
        df = -1 * correlation(self.open, self.volume, 10)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)
        
    # Alpha#7	 ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1* 1))
    # def alpha007(self):
    #     adv20 = sma(self.volume, 20)
    #     alpha = -1 * ts_rank(abs(delta(self.close, 7)), 60) * sign(delta(self.close, 7))
    #     alpha[adv20 >= self.volume] = -1
    #     return alpha
    def alpha007(self):
        adv20 = sma(self.volume, 20)  # 计算20日平均成交量
        alpha = -1 * ts_rank(abs(delta(self.close, 7)), 60) * sign(delta(self.close, 7))

        # 对齐索引
        adv20_aligned = adv20.reindex_like(self.volume)

        alpha[adv20_aligned >= self.volume] = -1
        return alpha

    
    # Alpha#8	 (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)),10))))
    def alpha008(self):
        return -1 * (rank(((ts_sum(self.open, 5) * ts_sum(self.returns, 5)) -
                        delay((ts_sum(self.open, 5) * ts_sum(self.returns, 5)), 10))))
        
    # Alpha#9	 ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ?delta(close, 1) : (-1 * delta(close, 1))))
    def alpha009(self):
        delta_close = delta(self.close, 1)
        cond_1 = ts_min(delta_close, 5) > 0
        cond_2 = ts_max(delta_close, 5) < 0
        alpha = -1     * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return alpha
        
    # Alpha#10	 rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0)? delta(close, 1) : (-1 * delta(close, 1)))))
    def alpha010(self):
        delta_close = delta(self.close, 1)
        cond_1 = ts_min(delta_close, 4) > 0
        cond_2 = ts_max(delta_close, 4) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return alpha
        
    # Alpha#11	 ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) *rank(delta(volume, 3)))
    def alpha011(self):
        return ((rank(ts_max((self.vwap - self.close), 3)) + rank(ts_min((self.vwap - self.close), 3))) *rank(delta(self.volume, 3)))
        
    # Alpha#12	 (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    def alpha012(self):
        return sign(delta(self.volume, 1)) * (-1 * delta(self.close, 1))

    # Alpha#13	 (-1 * rank(covariance(rank(close), rank(volume), 5)))
    def alpha013(self):
        return -1 * rank(covariance(rank(self.close), rank(self.volume), 5))
        
    # Alpha#14	 ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
    def alpha014(self):
        df = correlation(self.open, self.volume, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * rank(delta(self.returns, 3)) * df
    
    # Alpha#15	 (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
    def alpha015(self):
        df = correlation(rank(self.high), rank(self.volume), 3)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * ts_sum(rank(df), 3)
        
    # Alpha#16	 (-1 * rank(covariance(rank(high), rank(volume), 5)))
    def alpha016(self):
        return -1 * rank(covariance(rank(self.high), rank(self.volume), 5))
        
    # Alpha#17	 (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) *rank(ts_rank((volume / adv20), 5)))
    def alpha017(self):
        adv20 = sma(self.volume, 20)
        return -1 * (rank(ts_rank(self.close, 10)) *
                    rank(delta(delta(self.close, 1), 1)) *
                    rank(ts_rank((self.volume / adv20), 5)))
        
    # Alpha#18	 (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open,10))))
    def alpha018(self):
        df = correlation(self.close, self.open, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * (rank((stddev(abs((self.close - self.open)), 5) + (self.close - self.open)) +
                        df))
        
    # Alpha#19	 ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns,250)))))
    def alpha019(self):
        return ((-1 * sign((self.close - delay(self.close, 7)) + delta(self.close, 7))) *
                (1 + rank(1 + ts_sum(self.returns, 250))))
        
    # Alpha#20	 (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open -delay(low, 1))))
    def alpha020(self):
        return -1 * (rank(self.open - delay(self.high, 1)) *
                    rank(self.open - delay(self.close, 1)) *
                    rank(self.open - delay(self.low, 1)))

    # Alpha#21	 ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close,2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume /adv20) == 1)) ? 1 : (-1 * 1))))
    def alpha021(self):
        cond_1 = sma(self.close, 8) + stddev(self.close, 8) < sma(self.close, 2)
        cond_2 = sma(self.volume, 20) / self.volume < 1
        alpha = pd.DataFrame(np.ones_like(self.close), index=self.close.index
                            )
    #        alpha = pd.DataFrame(np.ones_like(self.close), index=self.close.index,
    #                             columns=self.close.columns)
        alpha[cond_1 | cond_2] = -1
        return alpha
        
    # Alpha#22	 (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
    def alpha022(self):
        df = correlation(self.high, self.volume, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * delta(df, 5) * rank(stddev(self.close, 20))

    # Alpha#23	 (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
#     def alpha023(self):
#         cond = sma(self.high, 20) < self.high
#         alpha = pd.DataFrame(np.zeros_like(self.close),index=self.close.index,columns=['close'])
# #        alpha.at[cond,'close'] = -1 * delta(self.high, 2).fillna(value=0)
#         alpha.loc[cond, 'close'] = -1 * delta(self.high, 2).fillna(value=0)
#         return alpha
    def alpha023(self):
        sma_high = sma(self.high, 20).reindex_like(self.high)  # 确保对齐索引
        cond = sma_high < self.high

        alpha = pd.DataFrame(np.zeros_like(self.close), index=self.close.index, columns=['close'])
        alpha.loc[cond, 'close'] = -1 * delta(self.high, 2).fillna(0)
        return alpha


    # Alpha#24	 ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) ||((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close,100))) : (-1 * delta(close, 3)))
    def alpha024(self):
        cond = delta(sma(self.close, 100), 100) / delay(self.close, 100) <= 0.05
        alpha = -1 * delta(self.close, 3)
        alpha[cond] = -1 * (self.close - ts_min(self.close, 100))
        return alpha
        
    # Alpha#25	 rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
    def alpha025(self):
        adv20 = sma(self.volume, 20)
        return rank(((((-1 * self.returns) * adv20) * self.vwap) * (self.high - self.close)))
        
    # Alpha#26	 (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
    def alpha026(self):
        df = correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * ts_max(df, 3)
    
    # Alpha#27	 ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)
    ###
    ## Some Error, still fixing!!
    def alpha027(self):
        alpha = rank((sma(correlation(rank(self.volume), rank(self.vwap), 6), 2) / 2.0))
        alpha[alpha > 0.5] = -1
        alpha[alpha <= 0.5]=1
        return alpha  
        
    # Alpha#28	 scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
    def alpha028(self):
        adv20 = sma(self.volume, 20)
        df = correlation(adv20, self.low, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return scale(((df + ((self.high + self.low) / 2)) - self.close))

    # Alpha#29	 (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1),5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
    def alpha029(self):
        return (ts_min(rank(rank(scale(log(ts_sum(rank(rank(-1 * rank(delta((self.close - 1), 5)))), 2))))), 5) +
                ts_rank(delay((-1 * self.returns), 6), 5))

    # Alpha#30	 (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) +sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
    def alpha030(self):
        delta_close = delta(self.close, 1)
        inner = sign(delta_close) + sign(delay(delta_close, 1)) + sign(delay(delta_close, 2))
        return ((1.0 - rank(inner)) * ts_sum(self.volume, 5)) / ts_sum(self.volume, 20)

    # Alpha#31	 ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 *delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
    def alpha031(self):
        adv20 = sma(self.volume, 20)
        df = correlation(adv20, self.low, 12).replace([-np.inf, np.inf], 0).fillna(value=0)         
        p1=rank(rank(rank(decay_linear((-1 * rank(rank(delta(self.close, 10)))).to_frame(), 10)))) 
        p2=rank((-1 * delta(self.close, 3)))
        p3=sign(scale(df))
        
        return p1.CLOSE+p2+p3

    # Alpha#32	 (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5),230))))
    def alpha032(self):
        return scale(((sma(self.close, 7) / 7) - self.close)) + (20 * scale(correlation(self.vwap, delay(self.close, 5),230)))
        
    # Alpha#33	 rank((-1 * ((1 - (open / close))^1)))
    def alpha033(self):
        return rank(-1 + (self.open / self.close))
        
        # Alpha#34	 rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))

    def alpha034(self):
        inner = stddev(self.returns, 2) / stddev(self.returns, 5)
        inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        return rank(2 - rank(inner) - rank(delta(self.close, 1)))

        # Alpha#35	 ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 -Ts_Rank(returns, 32)))

    def alpha035(self):
        return ((ts_rank(self.volume, 32) *
                (1 - ts_rank(self.close + self.high - self.low, 16))) *
                (1 - ts_rank(self.returns, 32)))
                
        # Alpha#36	 (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open- close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap,adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))

    def alpha036(self):
        adv20 = sma(self.volume, 20)
        return (((((2.21 * rank(correlation((self.close - self.open), delay(self.volume, 1), 15))) + (0.7 * rank((self.open- self.close)))) + (0.73 * rank(ts_rank(delay((-1 * self.returns), 6), 5)))) + rank(abs(correlation(self.vwap,adv20, 6)))) + (0.6 * rank((((sma(self.close, 200) / 200) - self.open) * (self.close - self.open)))))
        
        # Alpha#37	 (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))

    def alpha037(self):
        return rank(correlation(delay(self.open - self.close, 1), self.close, 200)) + rank(self.open - self.close)
        
        # Alpha#38	 ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))

    def alpha038(self):
        inner = self.close / self.open
        inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        return -1 * rank(ts_rank(self.open, 10)) * rank(inner)
    
        # Alpha#39	 ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 +rank(sum(returns, 250))))

    def alpha039(self):
        adv20 = sma(self.volume, 20)
        #return ((-1 * rank(delta(self.close, 7) * (1 - rank(decay_linear((self.volume / adv20).to_frame(), 9).CLOSE)))) *
        #        (1 + rank(sma(self.returns, 250))))
        
        return ((-1 * rank(delta(self.close, 7) * (1 - rank(decay_linear((self.volume / adv20).to_frame(name='CLOSE'), 9)['CLOSE'])))) *
                (1 + rank(sma(self.returns, 250))))

        # Alpha#40	 ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))

    def alpha040(self):
        return -1 * rank(stddev(self.high, 10)) * correlation(self.high, self.volume, 10)

        # Alpha#41	 (((high * low)^0.5) - vwap)

    def alpha041(self):
        return pow((self.high * self.low),0.5) - self.vwap
        
        # Alpha#42	 (rank((vwap - close)) / rank((vwap + close)))

    def alpha042(self):
        return rank((self.vwap - self.close)) / rank((self.vwap + self.close))
            
        # Alpha#43	 (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))

    def alpha043(self):
        adv20 = sma(self.volume, 20)
        return ts_rank(self.volume / adv20, 20) * ts_rank((-1 * delta(self.close, 7)), 8)

        # Alpha#44	 (-1 * correlation(high, rank(volume), 5))

    def alpha044(self):
        df = correlation(self.high, rank(self.volume), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * df

        # Alpha#45	 (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) *rank(correlation(sum(close, 5), sum(close, 20), 2))))

    def alpha045(self):
        df = correlation(self.close, self.volume, 2)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * (rank(sma(delay(self.close, 5), 20)) * df *
                    rank(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2)))
        
        # Alpha#46	 ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ?(-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 :((-1 * 1) * (close - delay(close, 1)))))

    def alpha046(self):
        inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)
        alpha = (-1 * delta(self.close))
        alpha[inner < 0] = 1
        alpha[inner > 0.25] = -1
        return alpha

        # Alpha#47	 ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) /5))) - rank((vwap - delay(vwap, 5))))

    def alpha047(self):
        adv20 = sma(self.volume, 20)
        return ((((rank((1 / self.close)) * self.volume) / adv20) * ((self.high * rank((self.high - self.close))) / (sma(self.high, 5) /5))) - rank((self.vwap - delay(self.vwap, 5))))
        
        # Alpha#48	 (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) *delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))
        
        
        # Alpha#49	 (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))

    def alpha049(self):
        inner = (((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10))
        alpha = (-1 * delta(self.close))
        alpha[inner < -0.1] = 1
        return alpha
        
        # Alpha#50	 (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))

    def alpha050(self):
        return (-1 * ts_max(rank(correlation(rank(self.volume), rank(self.vwap), 5)), 5))
        
        # Alpha#51	 (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))

    def alpha051(self):
        inner = (((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10))
        alpha = (-1 * delta(self.close))
        alpha[inner < -0.05] = 1
        return alpha
    
        # Alpha#52	 ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) -sum(returns, 20)) / 220))) * ts_rank(volume, 5))

    def alpha052(self):
        return (((-1 * delta(ts_min(self.low, 5), 5)) *
                rank(((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220))) * ts_rank(self.volume, 5))
        
        # Alpha#53	 (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))

    def alpha053(self):
        inner = (self.close - self.low).replace(0, 0.0001)
        return -1 * delta((((self.close - self.low) - (self.high - self.close)) / inner), 9)

        # Alpha#54	 ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))

    def alpha054(self):
        inner = (self.low - self.high).replace(0, -0.0001)
        return -1 * (self.low - self.close) * (self.open ** 5) / (inner * (self.close ** 5))

        # Alpha#55	 (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low,12)))), rank(volume), 6))

    def alpha055(self):
        divisor = (ts_max(self.high, 12) - ts_min(self.low, 12)).replace(0, 0.0001)
        inner = (self.close - ts_min(self.low, 12)) / (divisor)
        df = correlation(rank(inner), rank(self.volume), 6)
        return -1 * df.replace([-np.inf, np.inf], 0).fillna(value=0)

        # Alpha#56	 (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
        # This Alpha uses the Cap, however I have not acquired the data yet

    def alpha056(self):
        return (0 - (1 * (rank((sma(self.returns, 10) / sma(sma(self.returns, 2), 3))) * rank((self.returns * self.cap)))))
        
        # Alpha#57	 (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))

    def alpha057(self):
        return (0 - (1 * ((self.close - self.vwap) / decay_linear(rank(ts_argmax(self.close, 30)).to_frame(), 2).CLOSE)))
        
        # Alpha#58	 (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume,3.92795), 7.89291), 5.50322))
        
        # Alpha#59	 (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap *(1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))
        
        
        # Alpha#60	 (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) -scale(rank(ts_argmax(close, 10))))))

    def alpha060(self):
        divisor = (self.high - self.low).replace(0, 0.0001)
        inner = ((self.close - self.low) - (self.high - self.close)) * self.volume / divisor
        return - ((2 * scale(rank(inner))) - scale(rank(ts_argmax(self.close, 10))))
        
        # Alpha#61	 (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))

    # def alpha061(self):
    #     adv180 = sma(self.volume, 180)
    #     return (rank((self.vwap - ts_min(self.vwap, 16))) < rank(correlation(self.vwap, adv180, 18)))
    def alpha061(self):
        adv180 = sma(self.volume, 180)
        lhs = rank(self.vwap - ts_min(self.vwap, 16))
        rhs = rank(correlation(self.vwap, adv180, 18)).reindex_like(lhs)
        return lhs < rhs

        # Alpha#62	 ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) +rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)

    # def alpha062(self):
    #     adv20 = sma(self.volume, 20)
    #     return ((rank(correlation(self.vwap, sma(adv20, 22), 10)) < rank(((rank(self.open) +rank(self.open)) < (rank(((self.high + self.low) / 2)) + rank(self.high))))) * -1)

    def alpha062(self):
        adv20 = sma(self.volume, 20)
        lhs = rank(correlation(self.vwap, sma(adv20, 22), 10))
        left = rank(self.open) + rank(self.open)
        right = rank((self.high + self.low) / 2) + rank(self.high)
        # 对齐索引
        right = right.reindex_like(left)
        cond = left < right
        rhs = rank(cond).reindex_like(lhs)  # 再次确保 rhs 与 lhs 对齐
        return (lhs < rhs) * -1
  
        # Alpha#63	 ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237))- rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180,37.2467), 13.557), 12.2883))) * -1)
        
        
        # Alpha#64	 ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054),sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 -0.178404))), 3.69741))) * -1)

    # def alpha064(self):
    #     adv120 = sma(self.volume, 120)
    #     return ((rank(correlation(sma(((self.open * 0.178404) + (self.low * (1 - 0.178404))), 13),sma(adv120, 13), 17)) < rank(delta(((((self.high + self.low) / 2) * 0.178404) + (self.vwap * (1 -0.178404))), 3.69741))) * -1)
        
    def alpha064(self):
        adv120 = sma(self.volume, 120)
        x = (self.open * 0.178404) + (self.low * (1 - 0.178404))
        y = (((self.high + self.low) / 2) * 0.178404) + (self.vwap * (1 - 0.178404))

        lhs = rank(correlation(sma(x, 13), sma(adv120, 13), 17))
        rhs = rank(delta(y, 3.69741)).reindex_like(lhs)

        return (lhs < rhs) * -1

    # Alpha#65	 ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60,8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)
    # def alpha065(self):
    #     adv60 = sma(self.volume, 60)
    #     return ((rank(correlation(((self.open * 0.00817205) + (self.vwap * (1 - 0.00817205))), sma(adv60,9), 6)) < rank((self.open - ts_min(self.open, 14)))) * -1)
    def alpha065(self):
        adv60 = sma(self.volume, 60)
        x = (self.open * 0.00817205) + (self.vwap * (1 - 0.00817205))
        lhs = rank(correlation(x, sma(adv60, 9), 6))
        rhs = rank(self.open - ts_min(self.open, 14)).reindex_like(lhs)
        return (lhs < rhs) * -1

    # Alpha#66	 ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low* 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
    def alpha066(self):
        term1 = rank(decay_linear(delta(self.vwap, 4), 7))  # 两个整数窗口
        numerator = ((self.low * 0.96633) + (self.low * (1 - 0.96633))) - self.vwap
        denominator = self.open - ((self.high + self.low) / 2)

        term2 = ts_rank(
            decay_linear(numerator / denominator, 11), 7  # ts_rank 窗口为整数
        )

        return (term1 + term2) * -1


    # Alpha#67	 ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap,IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)
    # Alpha#68	 ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) <rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
    # def alpha068(self):
    #     adv15 = sma(self.volume, 15)

    #     term1 = ts_rank(
    #         correlation(rank(self.high), rank(adv15), 9),
    #         14
    #     )

    #     weighted_price = self.close * 0.518371 + self.low * (1 - 0.518371)
    #     term2 = rank(delta(weighted_price, 1)) 

    #     return ((term1 < term2) * -1)

        # Alpha#69	 ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412),4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416),9.0615)) * -1)
            
        # Alpha#70	 ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close,IndClass.industry), adv50, 17.8256), 17.9171)) * -1)
        
        
        # Alpha#71	 max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180,12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap +vwap)))^2), 16.4662), 4.4388))

    def alpha071(self):
        adv180 = sma(self.volume, 180)

        # 第一项
        c1 = correlation(
            ts_rank(self.close, 4),
            ts_rank(adv180, 12),
            18
        )
        d1 = decay_linear(c1, 4)
        term1 = ts_rank(d1, 15)

        # 第二项
        diff_squared = ((self.low + self.open) - 2 * self.vwap) ** 2
        d2 = decay_linear(rank(diff_squared), 16)
        term2 = ts_rank(d2, 5)

        return np.maximum(term1, term2)

        #return max(ts_rank(decay_linear(correlation(ts_rank(self.close, 3), ts_rank(adv180,12), 18).to_frame(), 4).CLOSE, 16), ts_rank(decay_linear((rank(((self.low + self.open) - (self.vwap +self.vwap))).pow(2)).to_frame(), 16).CLOSE, 4))
    
        # Alpha#72	 (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) /rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671),2.95011)))

    def alpha072(self):
        adv40 = sma(self.volume, 40)

        # 分子
        numerator = correlation(((self.high + self.low) / 2), adv40, 9)
        numerator = decay_linear(numerator, 10)
        numerator = rank(numerator)

        # 分母
        vwap_ranked = ts_rank(self.vwap, 4)
        volume_ranked = ts_rank(self.volume, 18)
        denominator = correlation(vwap_ranked, volume_ranked, 7)
        denominator = decay_linear(denominator, 3)
        denominator = rank(denominator)

        return numerator / denominator

        # Alpha#73	 (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)),Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open *0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)

    def alpha073(self):
        term1 = rank(decay_linear(delta(self.vwap, 5), 3))

        weighted_open_low = (self.open * 0.147155) + (self.low * (1 - 0.147155))
        delta_term = delta(weighted_open_low, 2)
        term2 = ts_rank(
            decay_linear((-1) * (delta_term / weighted_open_low), 3),
            17
        )

        return -1 * np.maximum(term1, term2)

        #return (max(rank(decay_linear(delta(self.vwap, 5).to_frame(), 3).CLOSE),ts_rank(decay_linear(((delta(((self.open * 0.147155) + (self.low * (1 - 0.147155))), 2) / ((self.open *0.147155) + (self.low * (1 - 0.147155)))) * -1).to_frame(), 3).CLOSE, 17)) * -1)
        
        # Alpha#74	 ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) <rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791)))* -1)

    def alpha074(self):
        adv30 = sma(self.volume, 30)
        return ((rank(correlation(self.close, sma(adv30, 37), 15)) <rank(correlation(rank(((self.high * 0.0261661) + (self.vwap * (1 - 0.0261661)))), rank(self.volume), 11)))* -1)
        
        # Alpha#75	 (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50),12.4413)))

    def alpha075(self):
        adv50 = sma(self.volume, 50)
        return (rank(correlation(self.vwap, self.volume, 4)) < rank(correlation(rank(self.low), rank(adv50),12)))
    
        # Alpha#76	 (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)),Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81,8.14941), 19.569), 17.1543), 19.383)) * -1)
        

        # Alpha#77	 min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)),rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))

    def alpha077(self):
        adv40 = sma(self.volume, 40)

        # 第一部分，直接表达式 rank(decay_linear(..., 20.0451))
        term1 = (((self.high + self.low) / 2) + self.high) - (self.vwap + self.high)
        p1 = rank(decay_linear(term1, 21))  # 向上取整

        # 第二部分，rank(decay_linear(correlation(...), ...))
        corr_val = correlation(((self.high + self.low) / 2), adv40, 4)  # ceil(3.1614)
        p2 = rank(decay_linear(corr_val, 6))  # ceil(5.64125)

        return np.minimum(p1, p2)

        #return min(rank(decay_linear(((((self.high + self.low) / 2) + self.high) - (self.vwap + self.high)).to_frame(), 20).CLOSE),rank(decay_linear(correlation(((self.high + self.low) / 2), adv40, 3).to_frame(), 6).CLOSE))
    
        # Alpha#78	 (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428),sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))

    def alpha078(self):
        adv40 = sma(self.volume, 40)
        return (rank(correlation(ts_sum(((self.low * 0.352233) + (self.vwap * (1 - 0.352233))), 20),ts_sum(adv40,20), 7)).pow(rank(correlation(rank(self.vwap), rank(self.volume), 6))))
        
        # Alpha#79	 (rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))),IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150,9.18637), 14.6644)))
        
        # Alpha#80	 ((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))),IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)
        
    
        # Alpha#81	 ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054),8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)

    def alpha081(self):
        adv10 = sma(self.volume, 10)
        return ((rank(log(product(rank((rank(correlation(self.vwap, ts_sum(adv10, 50),8)).pow(4))), 15))) < rank(correlation(rank(self.vwap), rank(self.volume), 5))) * -1)
        
        # Alpha#82	 (min(rank(decay_linear(delta(open, 1.46063), 14.8717)),Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) +(open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)
        
        
        # Alpha#83	 ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high -low) / (sum(close, 5) / 5)) / (vwap - close)))

    def alpha083(self):
        return ((rank(delay(((self.high - self.low) / (ts_sum(self.close, 5) / 5)), 2)) * rank(rank(self.volume))) / (((self.high -self.low) / (ts_sum(self.close, 5) / 5)) / (self.vwap - self.close)))
        
        # Alpha#84	 SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close,4.96796))

    def alpha084(self):
        return pow(ts_rank((self.vwap - ts_max(self.vwap, 15)), 21), delta(self.close,5))
        
        # Alpha#85	 (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30,9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595),7.11408)))

    def alpha085(self):
        adv30 = sma(self.volume, 30)
        return (rank(correlation(((self.high * 0.876703) + (self.close * (1 - 0.876703))), adv30,10)).pow(rank(correlation(ts_rank(((self.high + self.low) / 2), 4), ts_rank(self.volume, 10),7))))
        
        # Alpha#86	 ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open+ close) - (vwap + open)))) * -1)
    def alpha086(self):
        adv20 = sma(self.volume, 20)
        adv_sum = ts_sum(adv20, 15)  # ceil(14.7444)
        
        term1 = ts_rank(correlation(self.close, adv_sum, 6), 21)  # ceil(6.00049), 20.4195
        term2 = rank((self.open + self.close) - (self.vwap + self.open))

        # 对齐到 self.volume 的 index，不对 term1 term2 取交集
        base_index = self.volume.index
        term1 = term1.reindex(base_index)
        term2 = term2.reindex(base_index)

        return ((term1 < term2) * -1).replace({True: -1, False: 0}).astype(float)

        # Alpha#87	 (max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))),1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81,IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)
        
        
        # Alpha#88	 min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))),8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60,20.6966), 8.01266), 6.65053), 2.61957))

    def alpha088(self):
        adv60 = sma(self.volume, 60)

        term1_raw = (rank(self.open) + rank(self.low)) - (rank(self.high) + rank(self.close))
        term1 = rank(decay_linear(term1_raw, 8.06882))

        term2 = correlation(
            ts_rank(self.close, 8.44728),
            ts_rank(adv60, 20.6966),
            8.01266
        )
        term2 = ts_rank(decay_linear(term2, 6.65053), 2.61957)

        return np.minimum(term1, term2)
        #return min(rank(decay_linear(((rank(self.open) + rank(self.low)) - (rank(self.high) + rank(self.close))).to_frame(),8).CLOSE), ts_rank(decay_linear(correlation(ts_rank(self.close, 8), ts_rank(adv60,20.6966), 8).to_frame(), 7).CLOSE, 3))
        
        # Alpha#89	 (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10,6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap,IndClass.industry), 3.48158), 10.1466), 15.3012))
        # Alpha#90	 ((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40,IndClass.subindustry), low, 5.38375), 3.21856)) * -1)        
        # Alpha#91	 ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close,IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) -rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)
        # Alpha#92	 min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221),18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024),6.80584))

    def alpha092(self):
        adv30 = sma(self.volume, 30)

        cond = ((((self.high + self.low) / 2) + self.close) < (self.low + self.open))
        term1 = ts_rank(decay_linear(cond.astype(float), 14.7221), 18.8683)

        term2 = correlation(rank(self.low), rank(adv30), 7.58555)
        term2 = ts_rank(decay_linear(term2, 6.94024), 6.80584)

        return np.minimum(term1, term2)

        #return  min(ts_rank(decay_linear(((((self.high + self.low) / 2) + self.close) < (self.low + self.open)).to_frame(), 15).CLOSE,19), ts_rank(decay_linear(correlation(rank(self.low), rank(adv30), 8).to_frame(), 7).CLOSE,7))
    
        # Alpha#93	 (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81,17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 -0.524434))), 2.77377), 16.2664)))
        
        
        # Alpha#94	 ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap,19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)

    def alpha094(self):
        adv60 = sma(self.volume, 60)
        return ((rank((self.vwap - ts_min(self.vwap, 12))).pow(ts_rank(correlation(ts_rank(self.vwap,20), ts_rank(adv60, 4), 18), 3)) * -1))
    
        # Alpha#95	 (rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low)/ 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))

    def alpha095(self):
        adv40 = sma(self.volume, 40)

        term1 = rank(self.open - ts_min(self.open, 12))

        corr = correlation(
            sma(((self.high + self.low) / 2), 19),
            sma(adv40, 19),
            13
        )
        term2 = ts_rank(rank(corr).pow(5), 12)

        # 对齐两个序列到原始索引
        base_index = self.volume.index
        term1 = term1.reindex(base_index)
        term2 = term2.reindex(base_index)

        return ((term1 < term2) * -1).replace({True: -1, False: 0}).astype(float)

        # Alpha#96	 (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878),4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404),Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)

    def alpha096(self):
        adv60 = sma(self.volume, 60)

        term1 = correlation(rank(self.vwap), rank(self.volume), 3.83878)
        term1 = decay_linear(term1, 4.16783)
        term1 = ts_rank(term1, 8.38151)

        corr_term = correlation(
            ts_rank(self.close, 7.45404),
            ts_rank(adv60, 4.13242),
            3.65459
        )
        term2 = ts_argmax(corr_term, 12.6556)
        term2 = decay_linear(term2, 14.0365)
        term2 = ts_rank(term2, 13.4143)

        return (-1) * np.maximum(term1, term2)

        #return (max(ts_rank(decay_linear(correlation(rank(self.vwap), rank(self.volume).to_frame(), 4),4).CLOSE, 8), ts_rank(decay_linear(ts_argmax(correlation(ts_rank(self.close, 7),ts_rank(adv60, 4), 4), 13).to_frame(), 14).CLOSE, 13)) * -1)
        
        # Alpha#97	 ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))),IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low,7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)
        
        
        # Alpha#98	 (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) -rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571),6.95668), 8.07206)))

    def alpha098(self):
        adv5 = sma(self.volume, 5)
        adv15 = sma(self.volume, 15)

        # 左边部分：rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088))
        term1 = ts_sum(adv5, 27)  # ceil(26.4719)
        corr1 = correlation(self.vwap, term1, 5)  # ceil(4.58418)
        dec1 = decay_linear(corr1, 8)  # ceil(7.18088)
        r1 = rank(dec1)

        # 右边部分：rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571),6.95668), 8.07206))
        corr2 = correlation(rank(self.open), rank(adv15), 21)  # ceil(20.8187)
        argmin = ts_argmin(corr2, 9)  # ceil(8.62571)
        ts_r = ts_rank(argmin, 7)  # ceil(6.95668)
        dec2 = decay_linear(ts_r, 9)  # ceil(8.07206)
        r2 = rank(dec2)

        # 对齐索引
        base_index = self.volume.index
        r1 = r1.reindex(base_index)
        r2 = r2.reindex(base_index)

        return (r1 - r2).astype(float)

        # Alpha#99	 ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) <rank(correlation(low, volume, 6.28259))) * -1)

    def alpha099(self):
        adv60 = sma(self.volume, 60)

        term1 = correlation(
            ts_sum(((self.high + self.low) / 2), 20),
            ts_sum(adv60, 20),
            9
        )
        r1 = rank(term1)

        term2 = correlation(self.low, self.volume, 6)
        r2 = rank(term2)

        # 对齐两个 rank 序列
        base_index = self.volume.index
        r1 = r1.reindex(base_index)
        r2 = r2.reindex(base_index)

        return ((r1 < r2) * -1).astype(float)

        # Alpha#100	 (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high -close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) -scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))),IndClass.subindustry))) * (volume / adv20))))
        # Alpha#101	 ((close - open) / ((high - low) + .001))

    def alpha101(self):
        return (self.close - self.open) /((self.high - self.low) + 0.001)
     
# 加载你的 CSV 数据
data = pd.read_csv("train.csv")

# 重命名列名，使其与脚本中的变量名称匹配
data.rename(columns={
    '股票代码': 'STOCK_ID',
    '日期': 'DATE',
    '开盘': 'OPEN',
    '收盘': 'CLOSE',
    '最高': 'HIGH',
    '最低': 'LOW',
    '成交量': 'VOLUME',
    '成交额': 'AMOUNT',
    '涨跌幅': 'PCTCHANGE',
}, inplace=True)



# ========== Get Alpha Function ==========
def get_alpha(df):
    df = prepare_data(df)
    stock = Alphas(df)
    result = pd.DataFrame(index=df.index)
    result['alpha001'] = stock.alpha001()
    result['alpha002'] = stock.alpha002()
    result['alpha003'] = stock.alpha003()
    result['alpha004'] = stock.alpha004()
    result['alpha005'] = stock.alpha005()
    result['alpha006'] = stock.alpha006()
    result['alpha007'] = stock.alpha007()
    result['alpha008'] = stock.alpha008()
    result['alpha009'] = stock.alpha009()
    result['alpha010'] = stock.alpha010()
    result['alpha011'] = stock.alpha011()
    result['alpha012'] = stock.alpha012()
    result['alpha013'] = stock.alpha013()
    result['alpha014'] = stock.alpha014()
    result['alpha015'] = stock.alpha015()
    result['alpha016'] = stock.alpha016()
    result['alpha017'] = stock.alpha017()
    result['alpha018'] = stock.alpha018()
    result['alpha019'] = stock.alpha019()
    result['alpha020'] = stock.alpha020()
    result['alpha021'] = stock.alpha021()
    result['alpha022'] = stock.alpha022()
    result['alpha023'] = stock.alpha023()
    result['alpha024'] = stock.alpha024()
    result['alpha025'] = stock.alpha025()
    result['alpha026'] = stock.alpha026()
    result['alpha027'] = stock.alpha027()
    result['alpha028'] = stock.alpha028()
    result['alpha029'] = stock.alpha029()
    result['alpha030'] = stock.alpha030()
    result['alpha031'] = stock.alpha031()
    result['alpha032'] = stock.alpha032()
    result['alpha033'] = stock.alpha033()
    result['alpha034'] = stock.alpha034()
    result['alpha035'] = stock.alpha035()
    result['alpha036'] = stock.alpha036()
    result['alpha037'] = stock.alpha037()
    result['alpha038'] = stock.alpha038()
    result['alpha039'] = stock.alpha039()
    result['alpha040'] = stock.alpha040()
    result['alpha041'] = stock.alpha041()
    result['alpha042'] = stock.alpha042()
    result['alpha043'] = stock.alpha043()
    result['alpha044'] = stock.alpha044()
    result['alpha045'] = stock.alpha045()
    result['alpha046'] = stock.alpha046()
    result['alpha047'] = stock.alpha047()
    result['alpha049'] = stock.alpha049()
    result['alpha050'] = stock.alpha050()
    result['alpha051'] = stock.alpha051()
    result['alpha052'] = stock.alpha052()
    result['alpha053'] = stock.alpha053()
    result['alpha054'] = stock.alpha054()
    result['alpha055'] = stock.alpha055()
    #result['alpha056'] = stock.alpha056()
    result['alpha057'] = stock.alpha057()
    result['alpha060'] = stock.alpha060()
    result['alpha061'] = stock.alpha061()
    result['alpha062'] = stock.alpha062()
    result['alpha064'] = stock.alpha064()
    result['alpha065'] = stock.alpha065()
    result['alpha066'] = stock.alpha066()
    #result['alpha068'] = stock.alpha068()
    result['alpha071'] = stock.alpha071()
    result['alpha072'] = stock.alpha072()
    result['alpha073'] = stock.alpha073()
    result['alpha074'] = stock.alpha074()
    result['alpha075'] = stock.alpha075()
    result['alpha077'] = stock.alpha077()
    result['alpha078'] = stock.alpha078()
    result['alpha081'] = stock.alpha081()
    result['alpha083'] = stock.alpha083()
    result['alpha084'] = stock.alpha084()
    result['alpha085'] = stock.alpha085()
    result['alpha086'] = stock.alpha086()
    result['alpha088'] = stock.alpha088()
    result['alpha092'] = stock.alpha092()
    result['alpha094'] = stock.alpha094()
    result['alpha095'] = stock.alpha095()
    result['alpha096'] = stock.alpha096()
    result['alpha098'] = stock.alpha098()
    result['alpha099'] = stock.alpha099()
    result['alpha101'] = stock.alpha101()
    return result.reset_index()


# 计算 Alpha 因子
calculated_factors = get_alpha(data)
calculated_factors.to_csv('alpha001-101.csv', index=False)
print("Alpha 因子计算完成，结果已保存")


