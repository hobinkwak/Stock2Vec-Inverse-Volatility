import warnings

warnings.simplefilter('ignore')
from datetime import timedelta
from functools import reduce
from tqdm import tqdm
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from utils.embedding import Stock2Vec


def get_price_df(asset_list):
    print("데이터 다운로드 중...")
    price_df = pd.DataFrame()
    for asset in tqdm(asset_list):
        data = fdr.DataReader(asset)[['Close']]
        price_df = pd.concat([price_df, data], axis=1)
    price_df.columns = asset_list
    price_df = price_df.dropna()
    print("데이터 다운로드 완료...")
    return price_df


def get_rebal_dates(price_df, start_year='2008'):
    price_df['month'] = price_df.index.month
    price_df['year'] = price_df.index.year
    rebal_dates = price_df.drop_duplicates(subset=['year', 'month'], keep='last').loc[start_year:].index
    return rebal_dates


def weights_by_stock2vec(rtn_df, rebal_dates, optimize_func, days=180, wv_n_cluster=5, wv_size=100, wv_window=10,
                         wv_min_count=0,
                         wv_skipgram=True, lb=0.1, ub=0.3):
    weights = pd.DataFrame()
    for i in tqdm(range(len(rebal_dates))):
        sub_rtn_df = rtn_df.loc[rebal_dates[i] - timedelta(days=days): rebal_dates[i] - timedelta(days=1)]
        sv = Stock2Vec(sub_rtn_df.copy(), wv_n_cluster)
        df = sv.make_rtn_data()
        df = sv.sort_by_rtn(df)
        sv.train_n_save_word2vec(df, size=wv_size, window=wv_window, min_count=wv_min_count, skipgram=wv_skipgram)
        vectors = sv.get_sg_vectors()
        clusters = sv.kmeans_clustering(vectors)
        result = sv.extract_ticker(clusters)
        final_weights = optimize_func(sub_rtn_df, result, lb=lb, ub=ub)
        weights = pd.concat([weights, pd.DataFrame(final_weights.reshape(1, rtn_df.shape[-1]), index=[rebal_dates[i]],
                                                   columns=rtn_df.columns)])
    return weights


def weights_by_vanilla(rtn_df, rebal_dates, mode, days=180):
    assert mode in ['equal', 'inverse'], "mode는 equal (동일가중), inverse (inverse volatility)만 지원"
    weights = pd.DataFrame()
    for i in tqdm(range(len(rebal_dates))):
        sub_rtn_df = rtn_df.loc[rebal_dates[i] - timedelta(days=days): rebal_dates[i] - timedelta(days=1)]
        if mode == 'inverse':
            final_weights = ((1 / sub_rtn_df.std()) / (1 / sub_rtn_df.std()).sum()).values
        if mode == 'equal':
            final_weights = np.repeat(1 / rtn_df.shape[-1], rtn_df.shape[-1])
        weights = pd.concat([weights,
                             pd.DataFrame(final_weights.reshape(1, rtn_df.shape[-1]),
                                          index=[rebal_dates[i]], columns=rtn_df.columns)])
    return weights


def weights_by_optimize(rtn_df, rebal_dates, optimize_func, days=180, lb=0.05, ub=0.3):
    weights = pd.DataFrame()
    for i in tqdm(range(len(rebal_dates))):
        sub_rtn_df = rtn_df.loc[rebal_dates[i] - timedelta(days=days): rebal_dates[i] - timedelta(days=1)]
        final_weights = optimize_func(sub_rtn_df, lb=lb, ub=ub)
        weights = pd.concat([weights,
                             pd.DataFrame(final_weights.reshape(1, rtn_df.shape[-1]), index=[rebal_dates[i]],
                                          columns=rtn_df.columns)])
    return weights


def compute_portfolio_cum_rtn(price_df, weights):
    cum_rtn = 1
    individual_port_val_df_list = []

    prev_end_day = weights.index[0]
    for end_day in weights.index[1:]:
        sub_price_df = price_df.loc[prev_end_day:end_day]
        sub_asset_flow_df = sub_price_df / sub_price_df.iloc[0]

        weight_series = weights.loc[prev_end_day]
        indi_port_cum_rtn_series = (sub_asset_flow_df * weight_series) * cum_rtn

        individual_port_val_df_list.append(indi_port_cum_rtn_series)

        total_port_cum_rtn_series = indi_port_cum_rtn_series.sum(axis=1)
        cum_rtn = total_port_cum_rtn_series.iloc[-1]

        prev_end_day = end_day

    individual_port_val_df = reduce(lambda x, y: pd.concat([x, y.iloc[1:]]), individual_port_val_df_list)
    return individual_port_val_df
