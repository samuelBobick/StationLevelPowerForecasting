import numpy as np
import pandas as pd
import cvxpy as cp
from multiprocessing import Pool
from utils import (
    get_remaining_e_need,
    get_timestep_info
)
from constants.dcm import get_dcm_v

def argmin_u(
    u, 
    e_delivered, 
    J, 
    J_array, 
    current_peak_sch, 
    current_peak_reg, 
    constraints,
    z: list,
    v: np.ndarray,
    sub_df: pd.DataFrame,
    current_time: pd.Timestamp,
    running_peak: float,
    power_profiles: dict,
    prices: dict,
    delta_t, 
    power_rate, 
    flexibility_constant, 
    var_dim_constant,
    theta
):
        """
        Function to minimize charging cost. Flexible charging with variable power schedule

        Inputs:
            z: array where [tariff_flex, tariff_asap, tariff_overstay, leave = 1]
            v: array with softmax results [sm_c, sm_uc, sm_y] (sm_y = leave)
            sub_df: DataFrame containing rows of sessions_df that represent active \
                sessions at the time of optimization
            current_time: time of optimization
            running_peak: running peak power this billing cycle
            power_profiles: dictionary mapping dcosIds to power_profiles
            prices: dictionary mapping dcosIds to (sch_price, reg_price) tuples
        """
        obj = cp.Minimize(J)
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS, max_iters=10000, eps=1e-5)
        if prob.status == "optimal_inaccurate":
            # TODO: look into why this is happening
            print("WARNING: optimal solution found, but is inaccurate")
        elif prob.status != "optimal":
            raise Exception(f"Optimization failed with status {prob.status}")

        return (
            u.value,
            e_delivered.value,
            current_peak_sch.value,
            current_peak_reg.value,
            J,
            J_array,
        )



def parallel_grid_search(grid, sub_df, current_time, running_peak, power_profiles, prices, delta_t, power_rate, flexibility_constant, var_dim_constant, theta):
        # Prepare arguments for each grid point
        args_list = [
            (prices, grid[prices], sub_df, current_time, running_peak, power_profiles, delta_t, power_rate, flexibility_constant, var_dim_constant, theta)
            for prices in grid.keys()
        ]

        # Create a pool of workers
        with Pool() as pool:
            # Map the process_grid_point function to all grid points
            results = pool.map(process_grid_point, args_list)

        # Convert results list to dictionary
        grid_search_results = dict(results)
        
        return grid_search_results
    
def process_grid_point(args):
    prices, problem, sub_df, current_time, running_peak, power_profiles, delta_t, power_rate, flexibility_constant, var_dim_constant, theta = args
    z = [prices[0], prices[1], 1, 1]
    v = get_dcm_v(z, theta)
    u, e_delivered, J, J_array, current_peak_sch, current_peak_reg, constraints = problem

    (
        uk_flex,
        e_delivered,
        current_peak_sch,
        current_peak_reg,
        J,
        J_array,
    ) = argmin_u(u, e_delivered, J, J_array, current_peak_sch, current_peak_reg, constraints, z, v, sub_df, current_time, running_peak, power_profiles, prices, delta_t, power_rate, flexibility_constant, var_dim_constant, theta)

    return (prices[0], prices[1]), {
        "J": J.value[0] if isinstance(J.value, np.ndarray) else J.value,
        "J_arr": J_array,
        "u": uk_flex,
        "v": v,
        "current_peak_sch": current_peak_sch,
        "current_peak_reg": current_peak_reg,
    }