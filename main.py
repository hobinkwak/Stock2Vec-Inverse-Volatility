import warnings
import matplotlib.pyplot as plt
from utils.utils import *
from utils.objective import *

if __name__ == '__main__':
    warnings.simplefilter('ignore')
    asset_list = ['VTI', 'VEA', 'VWO', 'IAU', 'DBC', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
    price_df = get_price_df(asset_list)
    rebal_dates = get_rebal_dates(price_df)
    price_df = price_df.drop(['year','month'], axis=1)
    rtn_df = np.log(price_df / price_df.shift(1)).dropna()

    stock2vec_iv_mv = weights_by_stock2vec(rtn_df, rebal_dates, inverse_volatility_min_vol_optimize)
    stock2vec_db_iv = weights_by_stock2vec(rtn_df, rebal_dates, double_inverse_volatility_optimize)
    rp = weights_by_optimize(rtn_df, rebal_dates, rp_optimize)
    mv = weights_by_optimize(rtn_df, rebal_dates, min_vol_optimize)
    ms = weights_by_optimize(rtn_df, rebal_dates, max_sharpe_optimize)
    ew = weights_by_vanilla(rtn_df, rebal_dates, 'equal')
    iv = weights_by_vanilla(rtn_df, rebal_dates, 'inverse')

    cum_rtn1 = compute_portfolio_cum_rtn(price_df, stock2vec_iv_mv).sum(axis=1)
    cum_rtn2 = compute_portfolio_cum_rtn(price_df, stock2vec_db_iv).sum(axis=1)
    cum_rtn3 = compute_portfolio_cum_rtn(price_df, rp).sum(axis=1)
    cum_rtn4 = compute_portfolio_cum_rtn(price_df, mv).sum(axis=1)
    cum_rtn5 = compute_portfolio_cum_rtn(price_df, ms).sum(axis=1)
    cum_rtn6 = compute_portfolio_cum_rtn(price_df, ew).sum(axis=1)
    cum_rtn7 = compute_portfolio_cum_rtn(price_df, iv).sum(axis=1)


    plt.figure(figsize=(20, 10))
    cum_rtn1.plot(label='Stock2Vec Inverse Volatility + Min Vol')
    cum_rtn2.plot(label='Stock2Vec Double Inverse Volatility')
    cum_rtn3.plot(label='Risk Parity')
    cum_rtn4.plot(label='Min Vol')
    cum_rtn5.plot(label='Max Sharpe')
    cum_rtn6.plot(label='Equal Weights')
    cum_rtn7.plot(label='Inverse Volatility')
    plt.legend()
    plt.savefig('result/cum_return.png')
    plt.show()
