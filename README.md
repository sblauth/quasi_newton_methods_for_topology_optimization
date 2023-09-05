[![DOI](https://img.shields.io/badge/DOI-10.1007%2Fs00158--023--03653--2-blue)](https://doi.org/10.1007/s00158-023-03653-2)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7773404.svg)](https://doi.org/10.5281/zenodo.7773404)



The repository contains the source code for the numerical experiments considered
in [Quasi-Newton methods for topology optimization using a level-set method](https://doi.org/10.1007/s00158-023-03653-2) by Sebastian Blauth and Kevin Sturm.

To run the code, you have to install [cashocs](https://cashocs.readthedocs.io/)
first, which includes all necessary prerequisites. The results presented in this
repository have been obtained with version 2.0.0-alpha3 of cashocs (which uses FEniCS 2019.1).

The repository consists of the following test cases:

- A design identification problem constrained by a linear Poisson problem (named `linear_poisson_problem`) which is considered in Section 4.1 of the paper.

- A design identification problem constrained by a semilinear Poisson problem (named `semilinear_poisson_problem`) which is considered in Section 4.2 of the paper.

- Compliance minimization problems in linear elasticity: The cantilever (`linear_elasticity/cantilever`), bridges with single (`linear_elasticity/bridge_single`) and multiple (`linear_elasticity/bridge_multiple`) loads, which are investigated in Section 4.3 of the paper.

- Design optimization with Navier-Stokes flow: The pipe bend (`navier_stokes/pipe_bend`) and rugby ball (`navier_stokes/rugby_ball`) problems which are considered in Section 4.4 of the paper.


In each of the directories, there is a `main.py` file, which can be used to run the code. This file runs the entire benchmark, consisting of the so-called sphere combination, convex combination, gradient descent and BFGS methods, as presented in [Quasi-Newton methods for topology optimization using a level-set method](https://doi.org/10.1007/s00158-023-03653-2).

The file `visualization.py` generates the plots used in the paper. The repository is already initialized with the solutions obtained for the numerical examples in the paper, so that this can be run directly.

This software is citeable under the following DOI: [10.5281/zenodo.7773404](https://doi.org/10.5281/zenodo.7773404).

If you use the quasi-Newton methods for your work, please cite the paper

	Quasi-Newton methods for topology optimization using a level-set method
	Sebastian Blauth and Kevin Sturm
	Structural and Multidisciplinary Optimization, Volume 66, 2023
	https://doi.org/10.1007/s00158-023-03653-2

If you are using BibTeX, you can use the following entry:

	@Article{Blauth2023Quasi,
	  author   = {Blauth, Sebastian and Sturm, Kevin},
	  journal  = {Struct. Multidiscip. Optim.},
	  title    = {Quasi-{N}ewton methods for topology optimization using a level-set method},
	  year     = {2023},
	  issn     = {1615-147X,1615-1488},
	  number   = {9},
	  pages    = {203},
	  volume   = {66},
	  doi      = {10.1007/s00158-023-03653-2},
	  fjournal = {Structural and Multidisciplinary Optimization},
	  mrclass  = {99-06},
	  mrnumber = {4635978},
	}

