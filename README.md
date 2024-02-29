# NearestUnstableMatrix

[![Build Status](https://github.com/fph/NearestUnstableMatrix/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/fph/NearestUnstableMatrix/actions/workflows/CI.yml?query=branch%3Amain)

This is work-in-progress code for the solution of certain matrix nearness problems such as: what is the nearest matrix to a given matrix $A$ that has an eigenvalue in a certain closed region $\Omega \subseteq \mathbb{C}$? I.e., minimize
\[
  f(X) = \min_{X : \Lambda(X)\cap \mathbb{\Omega}\neq\emptyset} \norm{A-X}.
\]
The problem is reduced to optimization on manifolds and solved using the package Manopt.jl. See the files `src/example*.jl` for usage examples.

## Installation in dev mode

1. Install Julia.
2. Download or `git clone` the software to a folder of your choice.
3. Navigate to that folder, and open Julia with `julia --project=.`
4. Press `]` to obtain the blue `pkg>` prompt, then type `instantiate` and wait for download and precompilation of all dependencies.
5. The package is ready to use and develop! Remember to open Julia with `julia --project=.` every time.
