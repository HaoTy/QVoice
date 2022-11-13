import datetime

from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_finance.data_providers import RandomDataProvider
from qiskit_optimization import QuadraticProgram


# Quadratic objective with linear constraint
# Cost of evaluating constraints is O(n)
def get_random_instance(
    num_var: int, seed: int
) -> QuadraticProgram:
    # Generate expected return and covariance matrix from (random) time-series
    stocks = [("TICKER%s" % i) for i in range(num_var)]
    data = RandomDataProvider(
        tickers=stocks,
        start=datetime.datetime(2016, 1, 1),
        end=datetime.datetime(2016, 1, 30),
        seed=seed,
    )
    data.run()
    mu = data.get_period_return_mean_vector()
    sigma = data.get_period_return_covariance_matrix()

    return PortfolioOptimization(
        expected_returns=mu, covariances=sigma, risk_factor=0.5, budget=num_var // 2
    ).to_quadratic_program()
