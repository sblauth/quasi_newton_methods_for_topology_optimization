import sys

sys.path.insert(0, "../..")

import cashocs
from fenics import *
import utils

mu = 1e-2
alpha_in = 2.5 * mu / 1e2**2
alpha_out = 2.5 * mu * 1e2**2

cfg = cashocs.load_config("config.ini")
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(100, diagonal="crossed")

active = CompiledSubDomain(
    "x[0] >= tol && x[1] >= tol && x[0] <= 1 - tol && x[1] <= 1 - tol", tol=0.05
)
subdomains.set_all(0)
active.mark(subdomains, 1)
dx = Measure("dx", mesh, subdomain_data=subdomains)

vol_des = 0.8 * assemble(1 * dx(1))
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
psi_expr = Expression(
    "-pow(x[0] - 0.5, 2) - pow(x[1] - 0.5, 2) + pow(0.25, 2)", degree=2
)
psi.vector()[:] = interpolate(psi_expr, CG1).vector()[:]

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


def pressure_point(x, on_boundary):
    return near(x[0], 0) and near(x[1], 0)


bcs = cashocs.create_dirichlet_bcs(
    V.sub(0), Constant((1.0, 0.0)), boundaries, [1, 2, 3, 4]
)
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
    vol.vector()[0] = assemble(indicator_omega * dx(1))


sp = TrialFunction(CG1) * TestFunction(CG1) * dx(1)

for algorithm in utils.algorithm_list:
    psi.vector()[:] = interpolate(psi_expr, CG1).vector()[:]
    top = cashocs.TopologyOptimizationProblem(
        F,
        bcs,
        J,
        up,
        vq,
        psi,
        dJ_in,
        dJ_out,
        update_level_set,
        config=cfg,
        riesz_scalar_products=sp,
    )
    top.solve(algorithm=algorithm, rtol=0.0, atol=0.0, angle_tol=1.0, max_iter=250)
    utils.rename_files(algorithm, cfg)
