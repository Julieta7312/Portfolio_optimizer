import pandas as pd
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from functools import partial
from typing import Iterable, Union
import datetime as dt


p_minimize = partial(minimize, tol=1e-17, method='SLSQP', options={'disp': False, 'maxiter':1000})
def pf_opt(rets: pd.DataFrame, 
           rebal_dates: Iterable[dt.datetime],
           days: int = 250, 
           err: str = 'rp',
           cov_type: Union[None, str] = None,
           max_iters: int = 50) -> pd.DataFrame:
    
    ''' Return weights of the 'rp' (risk parity), 'mv' (minimum volatility) portfolios.
            
    Args:
        
        rets (required): pandas.core.frame.DataFrame
            DataFrame with ( index: Timestamps, columns: Asset names (str), values: Returns (float) )
        
        rebal_dates (required): iterable
            Rebalance dates, iterable containing Timestamps in the index of `rets`
            
        days (optional): int
            Days of Returns to compute sample covariance. 
            
        err (optional): str
            'rp': risk parity objective function
            'mv': minimum volatility objective function
        
        cov_type (optional): Not being used. 
            Can add some covariance estimation methods in the future.             
            
        max_iters (optional): int
            Number of iteration. (e.g. max_iters=50 -> Run maximum 50 * 1000 iteration for optimizer to converge)
    
    Returns:
    
        pd.DataFrame containing portfolio weights for given `rebal_dates`. 
    '''    
    
    n_asts = rets.shape[1]
    pf_risk_list = pd.DataFrame(data=np.zeros((len(rebal_dates), n_asts)),
                                index=rebal_dates,
                                columns=rets.columns)
    
    success_failure = []
    weights = None
    for date in tqdm(rebal_dates):
        
        ret = rets.loc[:date].iloc[-days:]
            
        if cov_type is None:
            cov_matrix = ret.cov()
        
        def risk_parity_error(weights):
            pf_risk = np.sqrt(weights.T @ (cov_matrix @ weights))
            risk_contribution = (weights.T * (weights.T @ cov_matrix))/pf_risk
            cont_row, cont_col = np.meshgrid(risk_contribution, risk_contribution)            
            return np.sum(np.square(cont_row - cont_col))
        
        def min_vol_error(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))        
        
        constraints = ({'type': 'eq'  , 'fun': lambda w: np.sum(w) - 1 },\
                       {'type': 'ineq', 'fun': lambda w: w })
        
        error = risk_parity_error if err == 'rp' else \
               (min_vol_error if err == 'mv' else min_vol_error )
               
        if weights is None:
            weights = np.ones(n_asts) / n_asts
        
        iteration = 0
        success = False
        while not success:
            
            if iteration > max_iters:
                break
            x0=weights if iteration==0 else opt_res.x 
            opt_res = p_minimize(fun=error,\
                                 x0=x0,\
                                 constraints=constraints)
            iteration += 1
            success = opt_res.success
        
        success_failure.append(opt_res.success)
        weights = opt_res.x
        
        pf_risk_list.loc[date, :] = list(opt_res.x)
    
    print(f"Optimization result: { sum(success_failure)/len(rebal_dates) }")
    
    return pf_risk_list



if __name__ == "__main__":
    
    from dateutil.relativedelta import relativedelta    
    
    end = dt.date.today()
    test_rets = pd.DataFrame(data=np.random.rand(1000, 10) / 10,
                             index=pd.date_range(start=end - relativedelta(days=999), end=end, freq='D'),
                             columns=[ f"asset{i}" for i in range(1, 11, 1) ])
    
    test_weights = pf_opt(rets=test_rets, 
                          rebal_dates=pd.Series(test_rets.index.to_period('M')).dt.to_timestamp()[249:])
    
    test_weights.sum(axis=1).plot()
    
    