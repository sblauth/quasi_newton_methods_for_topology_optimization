"""
Created on 22/09/2022, 15.51

@author: blauths
"""

import utils

result_dir = "."
cases = []

cases += [f"{result_dir}/linear_poisson_problem/results"]

cases += [f"{result_dir}/semilinear_poisson_problem/results"]

cases += [f"{result_dir}/linear_elasticity/cantilever/results"]
cases += [f"{result_dir}/linear_elasticity/bridge_single/results"]
cases += [f"{result_dir}/linear_elasticity/bridge_multiple/results"]

cases += [f"{result_dir}/navier_stokes/pipe_bend/results"]
cases += [f"{result_dir}/navier_stokes/rugby_ball/results"]

for case in cases:
    utils.visualize_case(case, "./img", show=False)
