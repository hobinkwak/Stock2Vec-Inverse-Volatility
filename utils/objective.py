import numpy as np
from scipy.optimize import minimize


def double_inverse_volatility_optimize(sub_rtn_df, result, lb=None, ub=None):
    ipv = []
    weights = []
    for i in result:
        w = ((1 / sub_rtn_df[i].std()) / (1 / sub_rtn_df[i].std()).sum()).values
        weights.append(w)
        port_vol = np.dot(np.dot(w, sub_rtn_df[i].cov()), w.T)
        inverse_port_vol = 1 / port_vol
        ipv.append(inverse_port_vol)
    ipv = np.array(ipv)
    ipv /= ipv.sum()
    final_weights = np.hstack([weights[i] * ipv[i] for i in range(len(result))])
    return final_weights


def inverse_volatility_min_vol_optimize(sub_rtn_df, result, lb, ub):
    def objective_func(weights, sub_rtn_df, result):
        sub_weights = []
        sub_port_vols = []
        sub_port_means = []
        sub_port_rtns = []
        for i in result:
            w = ((1 / sub_rtn_df[i].std()) / (1 / sub_rtn_df[i].std()).sum()).values
            sub_weights.append(w)
            sub_port_vol = np.sqrt(np.dot(np.dot(w, sub_rtn_df[i].cov()), w.T))
            sub_port_mean = np.dot(w, sub_rtn_df[i].mean())
            sub_port_rtn = sub_rtn_df[i].values * w.reshape(1, -1)
            sub_port_vols.append(sub_port_vol)
            sub_port_means.append(sub_port_mean)
            sub_port_rtns.append(sub_port_rtn.sum(axis=-1))
        ev = np.sqrt(np.dot(np.dot(weights, np.cov(np.array(sub_port_rtns), rowvar=True)), weights.T) * 252)
        return ev

    def constraint_sum(weights):
        return weights.sum() - 1

    def constraint_long(weights):
        return weights

    x0 = np.repeat(1 / len(result), len(result))
    constraints = ({'type': 'eq', 'fun': constraint_sum},
                   {'type': 'ineq', 'fun': constraint_long})
    options = {'maxiter': 1000, 'ftol': 1e-10}
    bounds = list(zip(np.repeat(lb, len(result)), np.repeat(ub, len(result))))
    optim = minimize(objective_func, args=(sub_rtn_df, result), bounds=bounds,
                     constraints=constraints, x0=x0, options=options)

    weights = optim.x
    asset_weights = []
    for idx, col in enumerate(result):
        w = ((1 / sub_rtn_df[col].std()) / (1 / sub_rtn_df[col].std()).sum()).values
        asset_weights.extend(list(w * weights[idx]))
    return np.array(asset_weights)


def rp_optimize(sub_rtn_df, lb, ub):
    def objective_func(weights, sub_rtn_df):
        sigma = np.sqrt(np.dot(np.dot(weights, sub_rtn_df.cov()), weights.T))
        mrc = np.dot(sub_rtn_df.cov(), weights.T) / sigma
        rc = mrc * weights
        rc = rc.reshape(-1, 1)
        rc_diff = rc - rc.T
        return np.square(rc_diff).sum()

    def constraint_sum(weights):
        return weights.sum() - 1

    def constraint_long(weights):
        return weights

    x0 = np.repeat(1 / sub_rtn_df.shape[-1], sub_rtn_df.shape[-1])
    constraints = ({'type': 'eq', 'fun': constraint_sum},
                   {'type': 'ineq', 'fun': constraint_long})
    options = {'maxiter': 1000, 'ftol': 1e-10}
    bounds = list(zip(np.repeat(lb, sub_rtn_df.shape[-1]), np.repeat(ub, sub_rtn_df.shape[-1])))
    optim = minimize(objective_func, args=(sub_rtn_df,), bounds=bounds,
                     constraints=constraints, x0=x0, options=options)

    return optim.x


def min_vol_optimize(sub_rtn_df, lb, ub):
    def objective_func(weights, sub_rtn_df):
        sigma = np.sqrt(np.dot(np.dot(weights, sub_rtn_df.cov()), weights.T) * 252)
        return sigma

    def constraint_sum(weights):
        return weights.sum() - 1

    def constraint_long(weights):
        return weights

    x0 = np.repeat(1 / sub_rtn_df.shape[-1], sub_rtn_df.shape[-1])
    constraints = ({'type': 'eq', 'fun': constraint_sum},
                   {'type': 'ineq', 'fun': constraint_long})
    options = {'maxiter': 1000, 'ftol': 1e-10}
    bounds = list(zip(np.repeat(lb, sub_rtn_df.shape[-1]), np.repeat(ub, sub_rtn_df.shape[-1])))
    optim = minimize(objective_func, args=(sub_rtn_df,), bounds=bounds,
                     constraints=constraints, x0=x0, options=options)

    return optim.x


def max_sharpe_optimize(sub_rtn_df, lb, ub):
    def objective_func(weights, sub_rtn_df):
        sigma = np.sqrt(np.dot(np.dot(weights, sub_rtn_df.cov()), weights.T) * 252)
        expected_rtn = np.dot(weights, sub_rtn_df.mean()) * 252
        return - expected_rtn / sigma

    def constraint_sum(weights):
        return weights.sum() - 1

    def constraint_long(weights):
        return weights

    x0 = np.repeat(1 / sub_rtn_df.shape[-1], sub_rtn_df.shape[-1])
    constraints = ({'type': 'eq', 'fun': constraint_sum},
                   {'type': 'ineq', 'fun': constraint_long})
    options = {'maxiter': 1000, 'ftol': 1e-10}
    bounds = list(zip(np.repeat(lb, sub_rtn_df.shape[-1]), np.repeat(ub, sub_rtn_df.shape[-1])))
    optim = minimize(objective_func, args=(sub_rtn_df,), bounds=bounds,
                     constraints=constraints, x0=x0, options=options)

    return optim.x
