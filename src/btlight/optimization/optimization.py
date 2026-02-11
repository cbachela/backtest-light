############################################################################
### QPMwP - OPTIMIZATION
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     18.01.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------



# Standard library imports
from abc import ABC, abstractmethod
from typing import Optional

# Third party imports
import numpy as np
import pandas as pd
import cvxpy as cp

# Local modules
from btlight.helper_functions import to_numpy
from btlight.estimation.covariance import Covariance
from btlight.estimation.expected_return import ExpectedReturn
from btlight.estimation.black_litterman import (
    bl_posterior_mu_sigma,
    generate_views_from_scores,
)
from btlight.optimization.optimization_data import OptimizationData
from btlight.optimization.constraints import Constraints
from btlight.optimization.quadratic_program import QuadraticProgram





# TODO:

# [ ] Add classes:
#    [x] MinVariance
#    [ ] MaxReturn
#    [ ] MaxSharpe
#    [ ] MaxUtility
#    [ ] RiskParity







class Objective():

    '''
    A class to handle the objective function of an optimization problem.

    Parameters:
    kwargs: Keyword arguments to initialize the coefficients dictionary. E.g. P, q, constant.
    '''

    def __init__(self, **kwargs):
        self.coefficients = kwargs

    @property
    def coefficients(self) -> dict:
        return self._coefficients

    @coefficients.setter
    def coefficients(self, value: dict) -> None:
        if isinstance(value, dict):
            self._coefficients = value
        else:
            raise ValueError('Input value must be a dictionary.')
        return None




class OptimizationParameter(dict):

    '''
    A class to handle optimization parameters.

    Parameters:
    kwargs: Additional keyword arguments to initialize the dictionary.
    '''

    def __init__(self, **kwargs):
        super().__init__(
            solver_name = 'cvxopt',
        )
        self.update(kwargs)



class Optimization(ABC):

    '''
    Abstract base class for optimization problems.

    Parameters:
    params (OptimizationParameter): Optimization parameters.
    kwargs: Additional keyword arguments.
    '''

    def __init__(self,
                 params: Optional[OptimizationParameter] = None,
                 constraints: Optional[Constraints] = None,
                 **kwargs):
        self.params = OptimizationParameter() if params is None else params
        self.params.update(**kwargs)
        self.constraints = Constraints() if constraints is None else constraints
        self.objective: Objective = Objective()
        self.results = {}

    @abstractmethod
    def set_objective(self, optimization_data: OptimizationData) -> None:
        raise NotImplementedError(
            "Method 'set_objective' must be implemented in derived class."
        )

    @abstractmethod
    def solve(self) -> None:

        # TODO:
        # Check consistency of constraints
        # self.check_constraints()

        # Get the coefficients of the objective function
        obj_coeff = self.objective.coefficients
        if 'P' not in obj_coeff.keys() or 'q' not in obj_coeff.keys():
            raise ValueError("Objective must contain 'P' and 'q'.")

        # Ensure that P and q are numpy arrays
        obj_coeff['P'] = to_numpy(obj_coeff['P'])
        obj_coeff['q'] = to_numpy(obj_coeff['q'])

        # if self.params.solver_name == "scip":
        #     self.solve_cvxpy()
        # else:
        self.solve_qpsolvers()
        return None

    def solve_qpsolvers(self) -> None:

        self.model_qpsolvers()
        self.model.solve()

        solution = self.model.results['solution']
        status = solution.found
        ids = self.constraints.ids
        # weights = pd.Series(solution.x[:len(ids)] if status else [None] * len(ids),
        #                     index=ids)
        weights = pd.Series(solution.x[:len(ids)], index=ids)

        self.results.update({
            'weights': weights.to_dict(),
            'status': status,
        })

        return None

    def model_qpsolvers(self) -> None:

        # constraints
        constraints = self.constraints
        GhAb = constraints.to_GhAb()
        lb = constraints.box['lower'].to_numpy() if constraints.box['box_type'] != 'NA' else None
        ub = constraints.box['upper'].to_numpy() if constraints.box['box_type'] != 'NA' else None

        # Solver settings
        solver_settings = dict(self.params)
        if 'solver_name' in solver_settings:
            solver_settings['solver'] = solver_settings.pop('solver_name')

        # Create the optimization model as a QuadraticProgram
        self.model = QuadraticProgram(
            P=self.objective.coefficients['P'],
            q=self.objective.coefficients['q'],
            G=GhAb['G'],
            h=GhAb['h'],
            A=GhAb['A'],
            b=GhAb['b'],
            lb=lb,
            ub=ub,
            **solver_settings
        )

        # Deal with turnover constraint or penalty (cannot have both)
        turnover_penalty = self.params.get('turnover_penalty')

        ## Turnover constraint
        tocon = self.constraints.l1.get('turnover')
        if tocon is not None and (turnover_penalty is None or turnover_penalty == 0):
            x_init = np.array(list(tocon['x0'].values()))
            self.model.linearize_turnover_constraint(x_init=x_init,
                                                     to_budget=tocon['rhs'])

        ## Turnover penalty
        if turnover_penalty is not None and turnover_penalty > 0:
            x_init = pd.Series(self.params.get('x_init')).to_numpy()
            self.model.linearize_turnover_objective(x_init=x_init,
                                                    turnover_penalty=turnover_penalty)

        return None

    # def solve_cvxpy(self):
    #     '''
    #     Solve the CVXPY problem using the SCIP solver and store results similarly to the Gurobi approach
    #    '''

    #     self.model_cvxpy()

    #     # Solve with SCIP
    #     self.model.solve(solver=cp.SCIP, verbose=False)

    #     # Extract solution and objective
    #     if self.model.status not in ["optimal", "optimal_inaccurate"]:
    #         raise ValueError(f"Optimization failed. Status: {self.problem.status}")

    #     # not standard to extract variables from model in cvxpy I guess
    #     # usually directly keeping track of the x in the previous function
    #     x_val = self.model.variables()[0].value
    #     obj_val = self.model.value
    #     status = self.model.status

    #     # Store results
    #     self.results = {
    #         'weights': x_val,
    #         'objective': obj_val,
    #         'status': status,
    #     }

    #     # Compute turnover (if x_init and auxiliary variable available)
    #     if hasattr(self, 'turnover') and callable(self.turnover):
    #         self.results['turnover'] = self.turnover()
    #     if hasattr(self, 'immediate_turnover') and callable(self.immediate_turnover):
    #         self.results['immediate_turnover'] = self.immediate_turnover()

    #     return

    # def model_cvxpy(self) -> None:

    #     P = self.objective.coefficients['P']
    #     q = self.objective.coefficients['q']

    #     lb = self.constraints.box.lower.to_numpy()
    #     ub = self.constraints.box.upper.to_numpy()

    #     N = len(self.constraints.selection)

    #     GhAb = self.constraints.to_GhAb()
    #     G, h = GhAb.get('G'), GhAb.get('h')
    #     A, b = GhAb.get('A'), GhAb.get('b')

    #     quadcon = self.constraints.quadratic
    #     x_init = np.array(self.params.x_init)
    #     transaction_cost = self.params.transaction_cost
    #     mip = self.constraints.mip
    #     l1 = self.constraints.l1

    #     # Decision variable
    #     x = cp.Variable(N, name='weights')
    #     constraints = [x >= lb, x <= ub]

    #     # Turnover cost
    #     if transaction_cost is not None:
    #         aux_turnover = cp.Variable(N)
    #         constraints += [
    #             x - aux_turnover <= x_init,
    #             x + aux_turnover >= x_init,
    #             aux_turnover >= 0
    #         ]
    #         obj = q @ x + 0.5 * cp.quad_form(x, P) + transaction_cost * cp.sum(aux_turnover)
    #     else:
    #         obj = q @ x + 0.5 * cp.quad_form(x, P)

    #     # Linear constraints
    #     if G is not None:
    #         constraints.append(G @ x <= h)
    #     if A is not None:
    #         constraints.append(A @ x == b)

    #     # Quadratic constraints
    #     for qc in quadcon.values():
    #         lhs = qc.q.T @ x + cp.quad_form(x, qc.Qc)
    #         if qc.sense == '<=':
    #             constraints.append(lhs <= qc.rhs)
    #         elif qc.sense == '>=':
    #             constraints.append(lhs >= qc.rhs)
    #         elif qc.sense == '=':
    #             constraints.append(lhs == qc.rhs)

    #     # Cardinality constraints (requires binary vars and a MIP-capable solver)
    #     if mip.cardinality is not None:
    #         b = cp.Variable(N, boolean=True)
    #         constraints += [x <= b, x >= np.maximum(lb.min(), 0.0001) * b]
    #         K = mip.cardinality.K
    #         if mip.cardinality.sense == '<=':
    #             constraints.append(cp.sum(b) <= K)
    #         elif mip.cardinality.sense == '>=':
    #             constraints.append(cp.sum(b) >= K)
    #         else:
    #             constraints += [cp.sum(b) <= K, cp.sum(b) >= K]

    #     # Min/Max asset count
    #     if mip.min_asset_count is not None:
    #         bmin = cp.Variable(N, boolean=True)
    #         constraints += [x <= bmin, x >= np.maximum(lb.min(), 0.0001) * bmin]
    #         constraints.append(cp.sum(bmin) >= mip.min_asset_count.K)

    #     if mip.max_asset_count is not None:
    #         bmax = cp.Variable(N, boolean=True)
    #         constraints += [x <= bmax, x >= np.maximum(lb.min(), 0.0001) * bmax]
    #         constraints.append(cp.sum(bmax) <= mip.max_asset_count.K)

    #     # Leverage constraint (||x||_1 <= rhs)
    #     if 'leverage' in l1:
    #         rhs = l1['leverage'].rhs
    #         rhs_neg = getattr(l1['leverage'], 'rhs_neg', None)
    #         aux_pos = cp.Variable(N)
    #         aux_neg = cp.Variable(N)
    #         constraints += [
    #             aux_pos - aux_neg == x,
    #             aux_pos >= 0,
    #             aux_neg >= 0,
    #             cp.sum(aux_pos + aux_neg) <= rhs
    #         ]
    #         if rhs_neg is not None:
    #             constraints.append(cp.sum(aux_neg) <= rhs_neg)

    #     # Turnover constraint
    #     if 'turnover' in l1:
    #         rhs = l1['turnover'].rhs
    #         x0 = np.array(l1['turnover'].x0)
    #         aux = cp.Variable(N)
    #         constraints += [
    #             x - aux <= x0,
    #             x + aux >= x0,
    #             aux >= 0,
    #             cp.sum(aux) <= rhs
    #         ]

    #     # Finalize problem
    #     self.model = cp.Problem(cp.Minimize(obj), constraints)



class EmptyOptimization(Optimization):
    '''
    Placeholder class for an optimization.
    This class is intended to be a placeholder and should not be used directly.
    '''

    def set_objective(self, optimization_data: OptimizationData) -> None:
        raise NotImplementedError(
            'EmptyOptimization is a placeholder and does not implement set_objective.'
        )

    def solve(self) -> None:
        raise NotImplementedError(
            'EmptyOptimization is a placeholder and does not implement solve.'
        )




class LeastSquares(Optimization):

    def __init__(self,
                 constraints: Optional[Constraints] = None,
                 covariance: Optional[Covariance] = None,
                 **kwargs):
        super().__init__(
            constraints=constraints,
            **kwargs
        )
        self.covariance = covariance

    def set_objective(self, optimization_data: OptimizationData) -> None:

        X = optimization_data['return_series']
        y = optimization_data['bm_series']
        if self.params.get('log_transform'):
            X = np.log(1 + X)
            y = np.log(1 + y)

        P = 2 * (X.T @ X)
        q = to_numpy(-2 * X.T @ y).reshape((-1,))
        constant = to_numpy(y.T @ y).item()

        l2_penalty = self.params.get('l2_penalty')
        if l2_penalty is not None and l2_penalty != 0:
            P += 2 * l2_penalty * np.eye(X.shape[1])

        self.objective = Objective(
            P=P,
            q=q,
            constant=constant
        )
        return None

    def solve(self) -> None:
        return super().solve()



class BlackLitterman(Optimization):

    def __init__(
        self,
        covariance: Optional[Covariance] = None,
        risk_aversion: float = 1,
        lambda_: float = 1,
        confidence: float = 1,
        tau_psi: Optional[float] = None,
        tau_omega: Optional[float] = None,
        view_gen_algo: str = 'absolute',
        # market_portfolio: str = 'from_optimization_data',
        signal_names: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            risk_aversion=risk_aversion,
            lambda_=lambda_,
            confidence=confidence,
            tau_psi=tau_psi,
            tau_omega=tau_omega,
            view_gen_algo=view_gen_algo,
            # market_portfolio=market_portfolio,
            signal_names=signal_names,
            **kwargs,
        )
        self.covariance = Covariance() if covariance is None else covariance

    def set_objective(self, optimization_data: OptimizationData) -> None:
        '''
        Sets the objective function for the optimization problem.
        
        Parameters:
        optimization_data: must contain return series (to compute the covariances) and scores.
        '''

        # Retrieve configuration parameters from the params attribute
        risk_aversion = self.params.get('risk_aversion')
        lambda_ = self.params.get('lambda_')
        confidence = self.params.get('confidence', 1)
        view_gen_algo = self.params.get('view_gen_algo')
        # market_portfolio = self.params.get('market_portfolio', 'from_optimization_data')
        signal_names = self.params.get('signal_names')

        # Calculate the covariance matrix
        self.covariance.estimate(
            X=optimization_data['return_series'],
            inplace=True,
        )

        # Parameters to scale uncertainty matrices
        n = self.covariance.matrix.shape[1]
        tau_psi = self.params.get('tau_psi') or 1/n
        tau_omega = self.params.get('tau_omega') or 1/n

        # Prior (market-implied) portfolio
        # if market_portfolio == 'from_optimization_data':
            # w_prior = optimization_data['cap_weights']
        # else:
        #     if market_portfolio == 'LeastSquares':
        #         mp = LeastSquares()
        #     else:
        #         raise ValueError(
        #             "market_portfolio must be one of "
        #             "['from_optimization_data', 'LeastSquares']"
        #         )
        #     mp.params.transaction_cost = None
        #     mp.constraints = copy.deepcopy(self.constraints)
        #     mp.params = copy.deepcopy(self.params)
        #     mp.set_objective(optimization_data=optimization_data)
        #     mp.solve()
        #     w_prior = pd.Series(mp.results['weights'])
        w_prior = optimization_data['cap_weights']

        # Calculate implied expected return
        mu_implied = risk_aversion * 2 * self.covariance.matrix @ w_prior

        # Extract signal scores
        scores = optimization_data['scores'][signal_names]

        # Construct the views
        P_tmp = {}
        q_tmp = {}
        for col in scores.columns:
            P_tmp[col], q_tmp[col] = generate_views_from_scores(
                scores=scores[col],
                mu_implied=mu_implied,
                method=view_gen_algo,
                scalefactor=1,
            )

        P = pd.concat(P_tmp, axis=0)
        q = pd.concat(q_tmp, axis=0)

        # Define the uncertainty of the views
        # Omega = pd.DataFrame(
        #     np.diag([tau_omega] * len(q)),
        #     index=q.index,
        #     columns=q.index
        # )
        # Alternatively:
        Omega = P @ self.covariance.matrix @ P.T * tau_omega
        Omega = pd.DataFrame(np.diag(np.diag(Omega)), P.index, P.index)

        # Define the uncertainty of the prior
        Psi = self.covariance.matrix * tau_psi

        # Compute the posterior expected return vector
        mu_posterior, sigma_posterior = bl_posterior_mu_sigma(
            mu_prior=mu_implied,
            covmat=self.covariance.matrix,
            P=P,
            q=q,
            Psi=Psi,
            Omega=Omega,
            confidence=confidence,
        )

        # Adjust transaction cost penalty
        # self.params.transaction_cost = self.params.transaction_cost * mu_posterior.std()

        # Set objective
        self.objective = Objective(
            mu_implied = mu_implied,
            q = mu_posterior * (-1),
            P = sigma_posterior * lambda_,
        )
        return None

    def solve(self) -> None:
        return super().solve()



class MeanVariance(Optimization):

    def __init__(self,
                 constraints: Optional[Constraints] = None,
                 covariance: Optional[Covariance] = None,
                 expected_return: Optional[ExpectedReturn] = None,
                 risk_aversion: float = 1,
                 **kwargs):
        super().__init__(
            constraints=constraints,
            risk_aversion=risk_aversion,
            **kwargs
        )
        self.covariance = Covariance() if covariance is None else covariance
        self.expected_return = ExpectedReturn() if expected_return is None else expected_return

    def set_objective(self, optimization_data: OptimizationData) -> None:
        X = optimization_data['return_series']
        covmat = self.covariance.estimate(X=X, inplace=False)
        mu = self.expected_return.estimate(X=X, inplace=False)
        self.objective = Objective(
            q = mu * -1,
            P = covmat * 2 * self.params['risk_aversion'],
        )
        return None

    def solve(self) -> None:
        return super().solve()



class MinVariance(Optimization):

    def __init__(self,
                 constraints: Optional[Constraints] = None,
                 covariance: Optional[Covariance] = None,
                 **kwargs):
        super().__init__(
            constraints=constraints,
            **kwargs
        )
        self.covariance = Covariance() if covariance is None else covariance

    def set_objective(self, optimization_data: OptimizationData) -> None:
        X = optimization_data['return_series']
        covmat = self.covariance.estimate(X=X, inplace=False)
        mu = np.zeros(X.shape[1])
        self.objective = Objective(
            q = mu ,
            P = covmat * 2,
        )
        return None

    def solve(self) -> None:
        if self.params.get('solver_name') == 'analytical':
            GhAb = self.constraints.to_GhAb()
            if GhAb['G'] is not None:
                raise ValueError(
                    'Analytical solution does not exist whith inequality constraints.'
                )
            A = GhAb['A']
            b = GhAb['b']
            # If b is scalar, convert it to a 1D array
            if isinstance(b, (int, float)):
                b = np.array([b])
            elif b.ndim == 0:
                b = np.array([b])

            P = self.objective.coefficients['P']
            P_inv = np.linalg.inv(P)

            AP_invA = A @ P_inv @ A.T
            if AP_invA.shape[0] > 1:
                AP_invA_inv = np.linalg.inv(AP_invA)
            else:
                AP_invA_inv = 1 / AP_invA
            x = pd.Series(P_inv @ A.T @ AP_invA_inv @ b,
                          index=self.constraints.ids)      
            self.results.update({
                'weights': x.to_dict(),
                'status': True,
            })
            return None
        else:
            return super().solve()


class ScoreVariance(Optimization):

    def __init__(self,
                 field: str,
                 constraints: Optional[Constraints] = None,
                 covariance: Optional[Covariance] = None,
                 risk_aversion: float = 1,
                 **kwargs):
        super().__init__(
            field=field,
            constraints=constraints,
            risk_aversion=risk_aversion,
            **kwargs,
        )
        self.covariance = Covariance() if covariance is None else covariance

    def set_objective(self, optimization_data: OptimizationData) -> None:

        # Arguments
        risk_aversion = self.params.get('risk_aversion')
        field = self.params.get('field')
        if field is None:
            raise ValueError('Field must be specified.')

        # Extract the scores from the optimization data
        scores = optimization_data['scores'][field]

        # Create quadratic part of the objective function
        # If risk aversion is not None and not equal to 0, use covariance matrix
        if risk_aversion is not None and risk_aversion != 0:
            P = self.covariance.estimate(
                X=optimization_data['return_series'],
                inplace=False
            ) * 2 * risk_aversion
        else:
            P = np.zeros(shape = (len(scores), len(scores)))
        self.objective = Objective(
            q = scores * (-1),
            P = P,
        )

        return None

    def solve(self) -> None:
        return super().solve()
