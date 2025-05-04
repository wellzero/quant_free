import numpy as np
import pandas as pd
import scipy.optimize as sco

def minimum_variance_weights(returns):
    """
    Calculate minimum variance portfolio weights
    
    Parameters:
    returns (DataFrame): DataFrame of asset returns
    
    Returns:
    Series: Portfolio weights
    """
    n = len(returns.columns)
    
    # Calculate covariance matrix
    cov_matrix = returns.cov()
    
    # Define optimization constraints
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    initial_weights = np.array([1/n] * n)
    
    # Define objective function (portfolio variance)
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    # Optimize
    result = sco.minimize(portfolio_variance, initial_weights, method='SLSQP',
                          bounds=bounds, constraints=constraints)
    
    # Return weights as a Series
    return pd.Series(result['x'], index=returns.columns)

def maximum_sharpe_weights(returns, risk_free_rate=0):
    """
    Calculate maximum Sharpe ratio portfolio weights
    
    Parameters:
    returns (DataFrame): DataFrame of asset returns
    risk_free_rate (float): Risk-free rate
    
    Returns:
    Series: Portfolio weights
    """
    n = len(returns.columns)
    
    # Calculate mean returns and covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    # Define optimization constraints
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    initial_weights = np.array([1/n] * n)
    
    # Define objective function (negative Sharpe ratio)
    def neg_sharpe_ratio(weights):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(portfolio_return - risk_free_rate) / portfolio_volatility
    
    # Optimize
    result = sco.minimize(neg_sharpe_ratio, initial_weights, method='SLSQP',
                          bounds=bounds, constraints=constraints)
    
    # Return weights as a Series
    return pd.Series(result['x'], index=returns.columns)

def risk_parity_weights(returns, risk_budget=None, max_iterations=100, tolerance=1e-8):
    """
    Calculate risk parity portfolio weights
    
    Parameters:
    returns (DataFrame): DataFrame of asset returns
    risk_budget (list, optional): Target risk contribution for each asset (equal by default)
    max_iterations (int): Maximum number of iterations for the algorithm
    tolerance (float): Convergence tolerance
    
    Returns:
    Series: Portfolio weights
    """
    n = len(returns.columns)
    
    # If risk budget is not provided, use equal risk allocation
    if risk_budget is None:
        risk_budget = np.array([1/n] * n)
    else:
        risk_budget = np.array(risk_budget)
        risk_budget = risk_budget / np.sum(risk_budget)  # Normalize
    
    # Calculate covariance matrix
    cov_matrix = returns.cov().values
    
    # Initial weights
    weights = np.array([1/n] * n)
    
    # Iterative algorithm for risk parity
    for _ in range(max_iterations):
        # Calculate portfolio volatility
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Calculate risk contribution of each asset
        marginal_risk = np.dot(cov_matrix, weights) / portfolio_vol
        risk_contribution = weights * marginal_risk
        risk_contribution_proportion = risk_contribution / np.sum(risk_contribution)
        
        # Check convergence
        error = np.sum(np.abs(risk_contribution_proportion - risk_budget))
        if error < tolerance:
            break
        
        # Update weights
        weights = weights * (risk_budget / risk_contribution_proportion)
        weights = weights / np.sum(weights)  # Normalize
    
    # Return weights as a Series
    return pd.Series(weights, index=returns.columns)

def efficient_frontier(returns, target_returns, risk_free_rate=0):
    """
    Calculate efficient frontier portfolios
    
    Parameters:
    returns (DataFrame): DataFrame of asset returns
    target_returns (list): List of target returns for the frontier
    risk_free_rate (float): Risk-free rate
    
    Returns:
    DataFrame: Efficient frontier portfolios with returns, volatility, and weights
    """
    n = len(returns.columns)
    
    # Calculate mean returns and covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    # Function to calculate portfolio volatility
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Function to calculate portfolio return
    def portfolio_return(weights):
        return np.sum(mean_returns * weights)
    
    # Constraints for optimization
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    
    # Calculate efficient frontier for each target return
    efficient_portfolios = []
    
    for target in target_returns:
        # Add constraint for target return
        target_constraint = {'type': 'eq', 'fun': lambda x: portfolio_return(x) - target}
        constraints_with_target = [constraints, target_constraint]
        
        # Initial weights
        initial_weights = np.array([1/n] * n)
        
        # Optimize to minimize volatility for the target return
        result = sco.minimize(portfolio_volatility, initial_weights, method='SLSQP',
                              bounds=bounds, constraints=constraints_with_target)
        
        # Calculate portfolio metrics
        weights = result['x']
        returns_val = portfolio_return(weights)
        volatility = portfolio_volatility(weights)
        sharpe = (returns_val - risk_free_rate) / volatility
        
        # Store results
        portfolio = {
            'Return': returns_val,
            'Volatility': volatility,
            'Sharpe': sharpe
        }
        
        # Add weights
        for i, asset in enumerate(returns.columns):
            portfolio[f'Weight_{asset}'] = weights[i]
        
        efficient_portfolios.append(portfolio)
    
    return pd.DataFrame(efficient_portfolios)

def optimize_portfolio(returns, objective='sharpe', risk_free_rate=0, target_return=None, target_risk=None):
    """
    Optimize portfolio based on different objectives
    
    Parameters:
    returns (DataFrame): DataFrame of asset returns
    objective (str): Optimization objective ('sharpe', 'min_variance', 'risk_parity', 'target_return', 'target_risk')
    risk_free_rate (float): Risk-free rate for Sharpe ratio calculation
    target_return (float): Target return for 'target_return' objective
    target_risk (float): Target risk for 'target_risk' objective
    
    Returns:
    dict: Optimized portfolio information
    """
    n = len(returns.columns)
    
    # Calculate mean returns and covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    # Function to calculate portfolio volatility
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Function to calculate portfolio return
    def portfolio_return(weights):
        return np.sum(mean_returns * weights)
    
    # Base constraints and bounds
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple((0, 1) for _ in range(n))
    initial_weights = np.array([1/n] * n)
    
    # Optimize based on objective
    if objective == 'sharpe':
        # Maximize Sharpe ratio
        def neg_sharpe_ratio(weights):
            portfolio_ret = portfolio_return(weights)
            portfolio_vol = portfolio_volatility(weights)
            return -(portfolio_ret - risk_free_rate) / portfolio_vol
        
        result = sco.minimize(neg_sharpe_ratio, initial_weights, method='SLSQP',
                              bounds=bounds, constraints=constraints)
        
    elif objective == 'min_variance':
        # Minimize variance
        result = sco.minimize(portfolio_volatility, initial_weights, method='SLSQP',
                              bounds=bounds, constraints=constraints)
        
    elif objective == 'risk_parity':
        # Risk parity (equal risk contribution)
        return {'weights': risk_parity_weights(returns)}
        
    elif objective == 'target_return':
        # Target return with minimum risk
        if target_return is None:
            raise ValueError("Target return must be provided for 'target_return' objective")
        
        # Add constraint for target return
        target_return_constraint = {'type': 'eq', 'fun': lambda x: portfolio_return(x) - target_return}
        constraints.append(target_return_constraint)
        
        result = sco.minimize(portfolio_volatility, initial_weights, method='SLSQP',
                              bounds=bounds, constraints=constraints)
        
    elif objective == 'target_risk':
        # Target risk with maximum return
        if target_risk is None:
            raise ValueError("Target risk must be provided for 'target_risk' objective")
        
        # Add constraint for target risk
        target_risk_constraint = {'type': 'eq', 'fun': lambda x: portfolio_volatility(x) - target_risk}
        constraints.append(target_risk_constraint)
        
        # Maximize return
        def neg_portfolio_return(weights):
            return -portfolio_return(weights)
        
        result = sco.minimize(neg_portfolio_return, initial_weights, method='SLSQP',
                              bounds=bounds, constraints=constraints)
    
    else:
        raise ValueError(f"Unknown objective: {objective}")
    
    # If we used scipy.optimize, extract the results
    if objective != 'risk_parity':
        weights = pd.Series(result['x'], index=returns.columns)
        portfolio_ret = portfolio_return(result['x'])
        portfolio_vol = portfolio_volatility(result['x'])
        sharpe = (portfolio_ret - risk_free_rate) / portfolio_vol
        
        return {
            'weights': weights,
            'return': portfolio_ret,
            'volatility': portfolio_vol,
            'sharpe': sharpe
        }