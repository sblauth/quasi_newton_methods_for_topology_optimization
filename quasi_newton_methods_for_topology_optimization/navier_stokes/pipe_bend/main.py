import sys

sys.path.insert(0, "../..")

import cashocs
from fenics import *
import numpy as np
import utils

mu = 1e-2
alpha_in = 2.5 * mu / 1e2**2
alpha_out = 2.5 * mu * 1e2**2

cfg = cashocs.load_config("config.ini")
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(100, diagonal="crossed")
vol_des = assemble(1 * dx) * 0.08 * np.pi
lambd = 1e4

v_elem = VectorElement("CG", mesh.ufl_cell(), 2)
p_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, v_elem * p_elem)
CG1 = FunctionSpace(mesh, "CG", 1)
DG0 = FunctionSpace(mesh, "DG", 0)
R = FunctionSpace(mesh, "R", 0)
vol = Function(R)

alpha = Function(DG0)
indicator_omega = Function(DG0)
psi = Function(CG1)
psi.vector()[:] = -1.0

up = Function(V)
u, p = split(up)
vq = Function(V)
v, q = split(vq)

F = (
    Constant(mu) * inner(grad(u), grad(v)) * dx
    + dot(grad(u) * u, v) * dx
    - p * div(v) * dx
    - q * div(u) * dx
    + alpha * dot(u, v) * dx
)

v_max = 1e0
v_in = Expression(
    ("(x[1] >= 0.7 && x[1] <= 0.9) ? v_max*(1 - 100*pow(x[1] - 0.8, 2)) : 0.0", "0.0"),
    degree=2,
    v_max=v_max,
)
v_out = Expression(
    ("0.0", "(x[0] >= 0.7 && x[0] <= 0.9) ? -v_max*(1 - 100*pow(x[0] - 0.8, 2)) : 0.0"),
    degree=2,
    v_max=v_max,
)


def pressure_point(x, on_boundary):
    return near(x[0], 0) and near(x[1], 0)


bcs = cashocs.create_dirichlet_bcs(V.sub(0), v_in, boundaries, 1)
bcs += cashocs.create_dirichlet_bcs(V.sub(0), v_out, boundaries, 3)
bcs += cashocs.create_dirichlet_bcs(V.sub(0), Constant((0.0, 0.0)), boundaries, [2, 4])
bcs += [DirichletBC(V.sub(1), Constant(0), pressure_point, method="pointwise")]

J = cashocs.IntegralFunctional(
    Constant(mu) * inner(grad(u), grad(u)) * dx
    + alpha * dot(u, u) * dx
    + Constant(lambd / 2) * pow(vol - Constant(vol_des), 2) * dx
)
dJ_in = Constant(alpha_in - alpha_out) * (dot(u, v) + dot(u, u)) + Constant(lambd) * (
    vol - Constant(vol_des)
)
dJ_out = Constant(alpha_in - alpha_out) * (dot(u, v) + dot(u, u)) + Constant(lambd) * (
    vol - Constant(vol_des)
)


def update_level_set():
    cashocs.interpolate_levelset_function_to_cells(psi, alpha_in, alpha_out, alpha)
    cashocs.interpolate_levelset_function_to_cells(psi, 1.0, 0.0, indicator_omega)
    vol.vector()[0] = assemble(indicator_omega * dx)


for algorithm in utils.algorithm_list:
    psi.vector()[:] = -1.0

    top = cashocs.TopologyOptimizationProblem(
        F, bcs, J, up, vq, psi, dJ_in, dJ_out, update_level_set, config=cfg
    )
    top.solve(algorithm=algorithm, rtol=0.0, atol=0.0, angle_tol=1.0, max_iter=500)
    utils.rename_files(algorithm, cfg)
