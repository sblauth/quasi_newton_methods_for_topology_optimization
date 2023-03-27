import sys

sys.path.insert(0, "..")

import cashocs
from fenics import *
import utils

cfg = cashocs.load_config("config.ini")

f_1 = 10.0
f_2 = 1.0
f_diff = f_1 - f_2

alpha_1 = 10.0
alpha_2 = 1.0
alpha_diff = alpha_1 - alpha_2

mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_box_mesh(
    n=96, start_x=-2, start_y=-2, end_x=2, end_y=2, diagonal="crossed"
)

V = FunctionSpace(mesh, "CG", 1)
DG0 = FunctionSpace(mesh, "DG", 0)

psi = Function(V)
psi.vector()[:] = 1.0

alpha = Function(DG0)
f = Function(DG0)


def create_desired_state():
    a = 4.0 / 5.0
    b = 2

    f_expr = Expression(
        "0.1 * ( "
        + "(sqrt(pow(x[0] - a, 2) + b * pow(x[1], 2)) - 1)"
        + "* (sqrt(pow(x[0] + a, 2) + b * pow(x[1], 2)) - 1)"
        + "* (sqrt(b * pow(x[0], 2) + pow(x[1] - a, 2)) - 1)"
        + "* (sqrt(b * pow(x[0], 2) + pow(x[1] + a, 2)) - 1)"
        + "- 0.001)",
        degree=1,
        a=a,
        b=b,
    )
    psi_des = interpolate(f_expr, V)

    cashocs.interpolate_levelset_function_to_cells(psi_des, alpha_1, alpha_2, alpha)
    cashocs.interpolate_levelset_function_to_cells(psi_des, f_1, f_2, f)

    y_des = Function(V)
    v = TestFunction(V)

    F = dot(grad(y_des), grad(v)) * dx + alpha * y_des * v * dx - f * v * dx
    bcs = cashocs.create_dirichlet_bcs(V, Constant(0.0), boundaries, [1, 2, 3, 4])

    cashocs.newton_solve(F, y_des, bcs, is_linear=True)

    return y_des


y_des = create_desired_state()

y = Function(V)
p = Function(V)
F = dot(grad(y), grad(p)) * dx + alpha * y * p * dx - f * p * dx
bcs = cashocs.create_dirichlet_bcs(V, Constant(0.0), boundaries, [1, 2, 3, 4])
J = cashocs.IntegralFunctional(Constant(0.5) * pow(y - y_des, 2) * dx)
dJ_in = Constant(alpha_diff) * y * p - Constant(f_diff) * p
dJ_out = Constant(alpha_diff) * y * p - Constant(f_diff) * p


def update_level_set():
    cashocs.interpolate_levelset_function_to_cells(psi, alpha_1, alpha_2, alpha)
    cashocs.interpolate_levelset_function_to_cells(psi, f_1, f_2, f)


for algorithm in utils.algorithm_list:
    psi.vector()[:] = 1.0

    top = cashocs.TopologyOptimizationProblem(
        F, bcs, J, y, p, psi, dJ_in, dJ_out, update_level_set, config=cfg
    )
    top.solve(algorithm=algorithm, rtol=0.0, atol=0.0, angle_tol=1.0, max_iter=500)

    utils.rename_files(algorithm, cfg)
