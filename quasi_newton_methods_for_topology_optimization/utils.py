import subprocess

import json
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pathlib


algorithm_list = (
    "sphere_combination",
    "convex_combination",
    "gradient_descent",
    "bfgs",
)


def visualize_case(case_name, location, show=False):
    plt.style.use("tableau-colorblind10")
    plt.rcParams.update({"font.size": 16})
    marker = ["s", "^", "o", "P"]
    linestyle = ["dotted", "dashed", "dashdot", "solid"]

    histories = []
    for algorithm in algorithm_list:
        with open(
            f"{case_name}/history_{algorithm}.json",
            "r",
        ) as file:
            histories.append(json.load(file))

    fig_cf, ax_cf = plt.subplots()
    fig_an, ax_an = plt.subplots()
    fig_grad, ax_grad = plt.subplots()

    for i, algorithm in enumerate(algorithm_list):
        ms = 6
        lw = 2

        cost_functional = histories[i]["cost_function_value"]
        domain = range(len(cost_functional))

        if len(cost_functional) > 125:
            lw = 2
            ms = 0

        ax_cf.plot(
            domain,
            cost_functional,
            label=algorithm,
            marker=marker[i],
            lw=lw,
            ms=ms,
            ls=linestyle[i],
        )
        ax_cf.semilogy()
        ax_cf.legend()
        ax_cf.set_xlabel("Iterations")
        ax_cf.set_ylabel("Cost function value")

        angle = np.array(histories[i]["angle"])
        domain = range(len(angle))

        ax_an.plot(
            domain,
            angle,
            label=algorithm,
            marker=marker[i],
            lw=lw,
            ms=ms,
            ls=linestyle[i],
        )
        ax_an.legend()
        ax_an.set_xlabel("Iterations")
        ax_an.set_ylabel("Angle [°]")
        ax_an.semilogy()
        ax_an.yaxis.set_major_formatter(ticker.ScalarFormatter())
        # ax_an.yaxis.set_minor_formatter(ticker.ScalarFormatter())
        lines = [pow(10, i) for i in range(3)]
        ax_an.hlines(lines, 0, len(angle), ls=":", color="k")
        # ax_an.set_ylim(bottom=0.5)

        gradient_norm = np.array(histories[i]["gradient_norm"])
        domain = range(len(gradient_norm))

        ax_grad.plot(
            domain,
            gradient_norm,
            label=algorithm,
            marker=marker[i],
            lw=lw,
            ms=ms,
            ls=linestyle[i],
        )
        ax_grad.legend()
        ax_grad.set_xlabel("Iterations")
        ax_grad.set_ylabel("Norm of Projected Gradient (relative)")
        ax_grad.semilogy()
        lines = [
            pow(10, -i)
            for i in range(0, int(-np.ceil(np.log10(ax_grad.get_ylim()[0]))) + 1)
        ]
        ax_grad.hlines(lines, 0, len(gradient_norm), ls=":", color="k")

    fig_cf.tight_layout()
    fig_an.tight_layout()
    fig_grad.tight_layout()

    case_split = case_name.split("/")
    if len(case_split) == 3:
        case_name = case_split[1]
    elif len(case_split) == 4:
        case_name = f"{case_split[1]}/{case_split[2]}"

    if not pathlib.Path(f"{location}/{case_name}").is_dir():
        pathlib.Path(f"{location}/{case_name}").mkdir(parents=True)

    fig_cf.savefig(
        f"{location}/{case_name}/cost_functional.png", dpi=250, bbox_inches="tight"
    )
    fig_an.savefig(f"{location}/{case_name}/angle.png", dpi=250, bbox_inches="tight")
    fig_grad.savefig(
        f"{location}/{case_name}/gradient_norm.png", dpi=250, bbox_inches="tight"
    )

    if not show:
        plt.close(fig_cf)
        plt.close(fig_an)
        plt.close(fig_grad)


def rename_files(algo: str, config) -> None:
    result_dir = config.get("Output", "result_dir")

    subprocess.run(["rm", f"{result_dir}/history_{algo}.json"])
    subprocess.run(
        [
            "mv",
            f"{result_dir}/history.json",
            f"{result_dir}/history_{algo}.json",
        ],
    )

    subprocess.run(["rm", f"{result_dir}/history_{algo}.txt"])
    subprocess.run(
        [
            "mv",
            f"{result_dir}/history.txt",
            f"{result_dir}/history_{algo}.txt",
        ],
    )
