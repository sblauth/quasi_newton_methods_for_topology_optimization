import sys

sys.path.insert(0, "../..")

import cashocs
from fenics import *
import utils

gamma = 120

E = 1.0
nu = 0.3
plane_stress = True

mu = E / (2.0 * (1.0 + nu))
lambd = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
if plane_stress:
    lambd = 2 * mu * lambd / (lambd + 2.0 * mu)

alpha_in = 1.0
alpha_out = 1e-3

cfg = cashocs.load_config("config.ini")

mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(
    48, length_x=2.0, length_y=1.2, diagonal="crossed"
)
V = VectorFunctionSpace(mesh, "CG", 1)
CG1 = FunctionSpace(mesh, "CG", 1)
DG0 = FunctionSpace(mesh, "DG", 0)

alpha = Function(DG0)
indicator_omega = Function(DG0)

psi = Function(CG1)
psi.vector()[:] = -1.0


def eps(u):
    return Constant(0.5) * (grad(u) + grad(u).T)


def sigma(u):
    return Constant(2.0 * mu) * eps(u) + Constant(lambd) * tr(eps(u)) * Identity(2)


class Delta1(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval(self, values, x):
        if near(x[0], 1.0) and near(x[1], 0.0):
            values[0] = 3.0 / mesh.hmax()
        else:
            values[0] = 0.0

    def value_shape(self):
        return ()


class Delta2(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval(self, values, x):
        if near(x[0], 0.5) and near(x[1], 0.0):
            values[0] = 3.0 / mesh.hmax()
        else:
            values[0] = 0.0

    def value_shape(self):
        return ()


class Delta3(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval(self, values, x):
        if near(x[0], 1.5) and near(x[1], 0.0):
            values[0] = 3.0 / mesh.hmax()
        else:
            values[0] = 0.0

    def value_shape(self):
        return ()


delta1 = Delta1()
delta2 = Delta2()
delta3 = Delta3()
g1 = delta1 * Constant((0.0, -1.0))
g2 = delta2 * Constant((0.0, -1.0))
g3 = delta3 * Constant((0.0, -1.0))

u1 = Function(V)
v1 = Function(V)
F1 = alpha * inner(sigma(u1), eps(v1)) * dx - dot(g1, v1) * ds(3)

u2 = Function(V)
v2 = Function(V)
F2 = alpha * inner(sigma(u2), eps(v2)) * dx - dot(g2, v2) * ds(3)

u3 = Function(V)
v3 = Function(V)
F3 = alpha * inner(sigma(u3), eps(v3)) * dx - dot(g3, v3) * ds(3)
F_list = [F1, F2, F3]
u_list = [u1, u2, u3]
v_list = [v1, v2, v3]


class BDRY(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], 0.0, DOLFIN_EPS) or near(x[0], 2.0, DOLFIN_EPS)) and near(
            x[1], 0.0
        )


bcs1 = [DirichletBC(V.sub(1), Constant(0.0), BDRY(), method="pointwise")]
bcs2 = [DirichletBC(V.sub(1), Constant(0.0), BDRY(), method="pointwise")]
bcs3 = [DirichletBC(V.sub(1), Constant(0.0), BDRY(), method="pointwise")]
bcs_list = [bcs1, bcs2, bcs3]

J = cashocs.IntegralFunctional(
    alpha
    * (
        inner(sigma(u1), eps(u1))
        + inner(sigma(u2), eps(u2))
        + inner(sigma(u3), eps(u3))
    )
    * dx
    + Constant(gamma) * indicator_omega * dx
)

kappa = (lambd + 3.0 * mu) / (lambd + mu)
r_in = alpha_out / alpha_in
r_out = alpha_in / alpha_out

dJ_in = (
    Constant(alpha_in * (r_in - 1.0) / (kappa * r_in + 1.0) * (kappa + 1.0) / 2.0)
    * (
        Constant(2.0)
        * (
            inner(sigma(u1), eps(u1))
            + inner(sigma(u2), eps(u2))
            + inner(sigma(u3), eps(u3))
        )
        + Constant((r_in - 1.0) * (kappa - 2.0) / (kappa + 2 * r_in - 1.0))
        * (
            tr(sigma(u1)) * tr(eps(u1))
            + tr(sigma(u2)) * tr(eps(u2))
            + tr(sigma(u3)) * tr(eps(u3))
        )
    )
) + Constant(gamma)
dJ_out = (
    Constant(-alpha_out * (r_out - 1.0) / (kappa * r_out + 1.0) * (kappa + 1.0) / 2.0)
    * (
        Constant(2.0)
        * (
            inner(sigma(u1), eps(u1))
            + inner(sigma(u2), eps(u2))
            + inner(sigma(u3), eps(u3))
        )
        + Constant((r_out - 1.0) * (kappa - 2.0) / (kappa + 2 * r_out - 1.0))
        * (
            tr(sigma(u1)) * tr(eps(u1))
            + tr(sigma(u2)) * tr(eps(u2))
            + tr(sigma(u3)) * tr(eps(u3))
        )
    )
) + Constant(gamma)


def update_level_set():
    cashocs.interpolate_levelset_function_to_cells(psi, alpha_in, alpha_out, alpha)
    cashocs.interpolate_levelset_function_to_cells(psi, 1.0, 0.0, indicator_omega)


for algorithm in utils.algorithm_list:
    psi.vector()[:] = -1.0
    top = cashocs.TopologyOptimizationProblem(
        F_list,
        bcs_list,
        J,
        u_list,
        v_list,
        psi,
        dJ_in,
        dJ_out,
        update_level_set,
        config=cfg,
    )
    top.solve(algorithm=algorithm, rtol=0.0, atol=0.0, angle_tol=1.5, max_iter=250)
    utils.rename_files(algorithm, cfg)
