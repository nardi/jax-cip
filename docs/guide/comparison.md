# Comparison with native SCIP

The suite in `tests/minlplib` solves each [MINLPLib](https://www.minlplib.org/)
instance twice. Native SCIP reads the OSiL file with its own reader and builds its own
expressions. JAX-CIP converts the same file into a [`Problem`][jaxcip.Problem] with one
JAX function per nonlinear row, and SCIP sees each term of a function as one opaque
expression. The suite then checks that:

- an optimal objective equals the known optimum from `minlplib.solu`, or native SCIP's
  when MINLPLib knows none,
- SCIP's own reading of the instance accepts JAX-CIP's primal solution,
- the dual bound never passes the known optimum,
- an infeasible instance yields no solution.

A timeout is recorded and not failed.

## Instances

The full run covers the 471 instances with a known optimum or a proof of infeasibility,
at most 60 variables and no semicontinuous variables or SOS constraints. The quick tier in `tests/minlplib/quick.txt` holds the 235
of them that both solvers finish in about a second, and it runs on every push.

## Results

Each solve had a 60 second limit, with four solves at a time on a four-core machine and
SCIP 10.1. Every optimum that JAX-CIP reports equals the known optimum. Native SCIP
reports a wrong optimum on `ex1252`, 134471.56 against the known 128893.74, which
JAX-CIP finds.

JAX-CIP solves 331 of the 427 instances that native SCIP solves, and 7 that native SCIP
does not finish, such as `alkylation` and `batch_nc`. Over the instances both solve, the
median JAX-CIP run takes 2.4 times as long and 1.24 times as many nodes. The prototype
solved 202 of the 236 instances of a smaller selection with a median time ratio of 2.0.
The times leave out building the programs, since `solving_seconds` counts SCIP's solve
only.

Most of the instances that only native SCIP solves time out with JAX-CIP. SCIP detects
structure in its own expressions that an opaque term hides, such as the convex norms of
`cvxnonsep_normcon*` and the complementarity products of `ex9_1_2` and `ex9_1_8`, whose
dual bounds stall without it.

## Known limitations

- `clay0203m` and `st_e05` stop with SCIP's LP error -6, which native SCIP does not hit
  on the same instances.
- `ghg_1veh` takes 411 seconds to build its programs, most of them for its 127 rows with
  products. The rows of the `cvxnonsep_nsig*` and `cvxnonsep_psig40` instances, with up to
  40 factors, take 21 to 29 seconds. The converter writes a product as one `jnp.prod`,
  which JAXormick relaxes in log₂ n levels. A chain of `*` takes minutes and gigabytes to
  compile at 30 factors.
- A problem with hundreds of distinct terms can reach Linux's limit on memory mappings,
  as [Performance](performance.md#build-time) describes.

## Running the comparison

`scripts/fetch_minlplib.py` downloads the OSiL files, `minlplib.solu` and
`instancedata.csv`. Point `JAXCIP_MINLPLIB_DIR` at its directory and set
`JAXCIP_MINLPLIB_RESULTS` to a file for the results, then run

```console
uv run pytest tests/minlplib --minlplib=full -n 4
uv run python benchmarks/minlplib_summary.py results.jsonl
```

The summary script prints the counts and median ratios of the results section and the
table below.

## Instance table

| Instance | Native | Seconds | JAX-CIP | Seconds | Nodes ratio |
| --- | --- | ---: | --- | ---: | ---: |
| `4stufen` | time_limit | 60.00 | time_limit | 60.00 | 5.27 |
| `alan` | optimal | 0.13 | optimal | 0.84 | 24.00 |
| `alkylation` | time_limit | 60.00 | optimal | 1.92 | 0.01 |
| `autocorr_bern20-03` | optimal | 0.00 | optimal | 1.68 | 601.00 |
| `autocorr_bern20-05` | optimal | 2.87 | time_limit | 60.00 | 4.37 |
| `autocorr_bern20-10` | optimal | 35.60 | time_limit | 60.00 | 0.32 |
| `autocorr_bern20-15` | time_limit | 60.00 | time_limit | 60.01 | 0.29 |
| `autocorr_bern25-03` | optimal | 0.01 | optimal | 1.76 | 596.00 |
| `autocorr_bern25-06` | time_limit | 60.00 | time_limit | 60.00 | 0.15 |
| `autocorr_bern25-13` | time_limit | 60.00 | time_limit | 60.01 | 0.98 |
| `autocorr_bern25-19` | time_limit | 60.00 | time_limit | 60.01 | 5.80 |
| `autocorr_bern25-25` | time_limit | 60.50 | time_limit | 60.08 | 1.00 |
| `autocorr_bern30-04` | optimal | 8.53 | time_limit | 60.00 | 5.54 |
| `autocorr_bern30-08` | time_limit | 60.00 | time_limit | 60.01 | 0.43 |
| `autocorr_bern35-04` | optimal | 16.68 | time_limit | 60.00 | 3.81 |
| `autocorr_bern35-09` | time_limit | 60.00 | time_limit | 60.00 | 0.87 |
| `autocorr_bern40-05` | time_limit | 60.00 | time_limit | 60.00 | 0.16 |
| `autocorr_bern45-05` | time_limit | 60.00 | time_limit | 60.00 | 0.16 |
| `ball_mk2_10` | optimal | 0.01 | optimal | 0.72 | 1497.00 |
| `ball_mk2_30` | optimal | 0.01 | time_limit | 60.01 | 161750.00 |
| `ball_mk3_10` | infeasible | 0.00 | infeasible | 3.79 | 642900.00 |
| `ball_mk3_20` | infeasible | 0.01 | time_limit | 60.21 | 33858.00 |
| `ball_mk3_30` | infeasible | 0.00 | time_limit | 60.00 | 14582.00 |
| `ball_mk4_05` | infeasible | 1.39 | time_limit | 60.00 | 99.17 |
| `ball_mk4_10` | time_limit | 60.00 | time_limit | 60.00 | 8.13 |
| `batch` | optimal | 1.26 | optimal | 6.25 | 120.54 |
| `batch_nc` | time_limit | 60.00 | optimal | 4.18 | 0.00 |
| `batchdes` | optimal | 0.18 | optimal | 0.04 | 0.50 |
| `blend029` | optimal | 3.48 | optimal | 12.02 | 3.48 |
| `blend146` | time_limit | 60.00 | time_limit | 60.00 | 0.43 |
| `blend480` | time_limit | 60.01 | time_limit | 60.01 | 1.08 |
| `cardqp_iqp` | time_limit | 60.00 | time_limit | 60.00 | 0.04 |
| `chance` | optimal | 0.03 | optimal | 0.08 | 3.00 |
| `chem` | time_limit | 60.00 | time_limit | 60.00 | 0.18 |
| `chenery` | optimal | 1.07 | optimal | 6.96 | 8.77 |
| `clay0203m` | optimal | 1.15 | error | - | - |
| `clay0204m` | optimal | 1.82 | optimal | 7.21 | 1.54 |
| `clay0303m` | optimal | 0.44 | optimal | 42.42 | 307.35 |
| `clay0304m` | optimal | 1.82 | time_limit | 60.00 | 37.77 |
| `crudeoil_lee4_07` | optimal | 52.10 | time_limit | 60.35 | 1.65 |
| `csched1a` | time_limit | 60.01 | optimal | 9.19 | 0.01 |
| `cvxnonsep_normcon20` | optimal | 0.13 | time_limit | 60.00 | 1837.20 |
| `cvxnonsep_normcon20r` | optimal | 0.35 | time_limit | 60.00 | 34014.75 |
| `cvxnonsep_normcon30` | optimal | 0.20 | time_limit | 60.00 | 990.67 |
| `cvxnonsep_normcon30r` | optimal | 0.13 | time_limit | 60.00 | 115211.00 |
| `cvxnonsep_normcon40` | optimal | 0.08 | time_limit | 60.00 | 129.71 |
| `cvxnonsep_nsig20` | optimal | 50.92 | time_limit | 60.00 | 0.15 |
| `cvxnonsep_nsig20r` | optimal | 0.34 | optimal | 0.60 | 4.50 |
| `cvxnonsep_nsig30` | time_limit | 60.00 | time_limit | 60.00 | 0.08 |
| `cvxnonsep_nsig30r` | optimal | 0.48 | optimal | 1.02 | 3.88 |
| `cvxnonsep_nsig40` | time_limit | 60.01 | time_limit | 60.00 | 0.05 |
| `cvxnonsep_pcon20` | optimal | 2.61 | time_limit | 60.00 | 877.15 |
| `cvxnonsep_pcon20r` | optimal | 0.25 | optimal | 0.22 | 6.00 |
| `cvxnonsep_pcon30` | time_limit | 60.00 | optimal | 19.82 | 13.19 |
| `cvxnonsep_pcon30r` | optimal | 0.22 | optimal | 0.39 | 3.25 |
| `cvxnonsep_pcon40` | optimal | 40.33 | time_limit | 60.00 | 93.45 |
| `cvxnonsep_pcon40r` | optimal | 0.38 | optimal | 0.30 | 1.00 |
| `cvxnonsep_psig20` | time_limit | 60.00 | time_limit | 62.02 | 0.00 |
| `cvxnonsep_psig20r` | optimal | 0.18 | time_limit | 60.00 | 1019.20 |
| `cvxnonsep_psig30` | time_limit | 60.00 | time_limit | 60.37 | 0.00 |
| `cvxnonsep_psig40` | time_limit | 60.00 | time_limit | 60.00 | 0.52 |
| `dispatch` | optimal | 0.01 | optimal | 52.39 | 248697.00 |
| `du-opt` | optimal | 15.69 | time_limit | 60.00 | 0.89 |
| `du-opt5` | optimal | 0.64 | time_limit | 60.00 | 64.41 |
| `etamac` | time_limit | 60.00 | time_limit | 60.00 | 3.41 |
| `ex1221` | optimal | 0.00 | optimal | 0.03 | 2.00 |
| `ex1222` | optimal | 0.00 | optimal | 0.10 | 1.00 |
| `ex1223` | optimal | 0.06 | optimal | 0.11 | 2.00 |
| `ex1223a` | optimal | 0.05 | optimal | 0.08 | 1.00 |
| `ex1223b` | optimal | 0.11 | optimal | 0.09 | 1.00 |
| `ex1224` | optimal | 0.02 | optimal | 0.33 | 2.00 |
| `ex1225` | optimal | 0.00 | optimal | 0.00 | 1.00 |
| `ex1226` | optimal | 0.01 | optimal | 0.01 | 1.00 |
| `ex1252` | optimal | 19.35 | optimal | 24.72 | 0.46 |
| `ex1252a` | optimal | 1.15 | time_limit | 60.00 | 20.34 |
| `ex1263a` | optimal | 0.10 | optimal | 0.11 | 26.00 |
| `ex1264a` | optimal | 0.02 | optimal | 0.11 | 58.00 |
| `ex1265a` | optimal | 0.04 | optimal | 0.17 | 41.00 |
| `ex1266a` | optimal | 0.05 | optimal | 0.14 | 62.00 |
| `ex14_1_1` | optimal | 0.01 | optimal | 0.46 | 37.00 |
| `ex14_1_2` | optimal | 0.02 | optimal | 2.78 | 177.00 |
| `ex14_1_3` | optimal | 0.01 | optimal | 0.29 | 79.00 |
| `ex14_1_4` | optimal | 0.01 | optimal | 0.81 | 23.00 |
| `ex14_1_5` | optimal | 0.01 | optimal | 0.73 | 53.00 |
| `ex14_1_6` | optimal | 0.03 | optimal | 0.21 | 599.00 |
| `ex14_1_7` | optimal | 22.57 | time_limit | 60.00 | 0.49 |
| `ex14_1_8` | optimal | 0.21 | optimal | 0.47 | 21.00 |
| `ex14_1_9` | optimal | 0.01 | optimal | 0.71 | 73.00 |
| `ex14_2_1` | optimal | 0.02 | optimal | 0.06 | 1.00 |
| `ex14_2_2` | optimal | 0.02 | optimal | 0.03 | 1.00 |
| `ex14_2_3` | optimal | 0.04 | optimal | 0.15 | 1.00 |
| `ex14_2_4` | optimal | 0.03 | optimal | 0.09 | 1.00 |
| `ex14_2_5` | optimal | 0.02 | optimal | 0.02 | 1.00 |
| `ex14_2_6` | optimal | 0.03 | optimal | 0.06 | 1.00 |
| `ex14_2_7` | optimal | 0.04 | optimal | 0.10 | 1.00 |
| `ex14_2_8` | optimal | 0.01 | optimal | 0.03 | 1.00 |
| `ex14_2_9` | optimal | 0.03 | optimal | 0.05 | 1.00 |
| `ex2_1_1` | optimal | 0.09 | optimal | 0.15 | 1.00 |
| `ex2_1_10` | optimal | 0.28 | optimal | 0.55 | 3.00 |
| `ex2_1_2` | optimal | 0.00 | optimal | 0.00 | 1.00 |
| `ex2_1_3` | optimal | 0.01 | optimal | 0.02 | 1.00 |
| `ex2_1_4` | optimal | 0.00 | optimal | 0.00 | 1.00 |
| `ex2_1_5` | optimal | 0.04 | optimal | 0.11 | 0.56 |
| `ex2_1_6` | optimal | 0.10 | optimal | 0.17 | 1.57 |
| `ex2_1_7` | optimal | 1.36 | optimal | 1.12 | 0.48 |
| `ex2_1_8` | optimal | 0.08 | optimal | 0.20 | 2.76 |
| `ex2_1_9` | optimal | 1.74 | optimal | 6.33 | 1.37 |
| `ex3_1_1` | optimal | 0.18 | optimal | 2.61 | 7.83 |
| `ex3_1_2` | optimal | 0.04 | optimal | 0.06 | 2.50 |
| `ex3_1_3` | optimal | 0.06 | optimal | 0.14 | 4.50 |
| `ex3_1_4` | optimal | 0.07 | optimal | 0.07 | 1.09 |
| `ex3pb` | optimal | 0.05 | optimal | 0.18 | 46.00 |
| `ex4` | optimal | 2.28 | optimal | 19.32 | 88.13 |
| `ex4_1_1` | optimal | 0.08 | optimal | 0.36 | 1.30 |
| `ex4_1_2` | optimal | 0.43 | time_limit | 60.00 | 7735.23 |
| `ex4_1_3` | optimal | 0.13 | optimal | 0.55 | 1.04 |
| `ex4_1_4` | optimal | 0.18 | optimal | 0.43 | 1.08 |
| `ex4_1_5` | optimal | 4.10 | time_limit | 60.00 | 15.15 |
| `ex4_1_6` | optimal | 0.16 | optimal | 0.51 | 1.00 |
| `ex4_1_7` | optimal | 0.34 | optimal | 0.17 | 1.31 |
| `ex4_1_8` | optimal | 0.02 | optimal | 0.10 | 11.00 |
| `ex4_1_9` | optimal | 0.17 | optimal | 0.64 | 1.00 |
| `ex5_2_2_case1` | optimal | 0.05 | optimal | 0.20 | 3.11 |
| `ex5_2_2_case2` | optimal | 0.04 | optimal | 0.23 | 4.31 |
| `ex5_2_2_case3` | optimal | 0.03 | optimal | 0.14 | 7.40 |
| `ex5_2_4` | optimal | 0.13 | optimal | 0.56 | 1.03 |
| `ex5_2_5` | time_limit | 60.00 | time_limit | 60.00 | 0.08 |
| `ex5_3_2` | optimal | 0.18 | optimal | 1.04 | 5.58 |
| `ex5_4_2` | optimal | 0.06 | optimal | 1.01 | 8.71 |
| `ex5_4_3` | optimal | 0.27 | optimal | 1.01 | 4.11 |
| `ex5_4_4` | optimal | 1.60 | time_limit | 60.00 | 5.91 |
| `ex6_1_1` | time_limit | 60.00 | time_limit | 60.00 | 5.35 |
| `ex6_1_3` | time_limit | 60.00 | time_limit | 60.01 | 3.59 |
| `ex6_1_4` | optimal | 0.24 | time_limit | 60.00 | 3042.96 |
| `ex6_2_14` | time_limit | 60.00 | optimal | 50.50 | 0.98 |
| `ex7_2_1` | time_limit | 60.00 | optimal | 1.39 | 0.00 |
| `ex7_2_2` | optimal | 0.19 | optimal | 1.08 | 1.18 |
| `ex7_2_4` | optimal | 1.43 | optimal | 8.06 | 1.91 |
| `ex7_3_1` | optimal | 0.03 | optimal | 0.04 | 1.00 |
| `ex7_3_2` | optimal | 0.03 | optimal | 0.07 | 7.00 |
| `ex7_3_3` | optimal | 0.08 | optimal | 0.31 | 39.00 |
| `ex7_3_4` | optimal | 0.47 | optimal | 6.32 | 4.24 |
| `ex7_3_6` | infeasible | 0.02 | infeasible | 1.57 | 202300.00 |
| `ex8_1_1` | optimal | 0.10 | optimal | 0.56 | 1.40 |
| `ex8_1_6` | optimal | 0.15 | time_limit | 60.01 | 6646.33 |
| `ex8_1_7` | optimal | 0.54 | optimal | 1.80 | 0.93 |
| `ex8_4_1` | optimal | 1.82 | time_limit | 60.00 | 8.99 |
| `ex9_1_1` | optimal | 0.01 | optimal | 0.01 | 1.00 |
| `ex9_1_2` | optimal | 0.03 | time_limit | 60.00 | 79065.00 |
| `ex9_1_4` | optimal | 0.00 | optimal | 0.00 | 1.00 |
| `ex9_1_5` | optimal | 0.06 | optimal | 0.13 | 1.00 |
| `ex9_1_8` | optimal | 0.02 | time_limit | 60.01 | 132810.00 |
| `ex9_2_2` | optimal | 0.03 | optimal | 1.40 | 49.00 |
| `ex9_2_3` | optimal | 0.26 | optimal | 0.67 | 0.74 |
| `ex9_2_4` | optimal | 0.09 | optimal | 0.71 | 51.00 |
| `ex9_2_5` | optimal | 0.10 | time_limit | 60.00 | 7931.52 |
| `ex9_2_6` | optimal | 0.05 | optimal | 0.76 | 95.00 |
| `ex9_2_7` | optimal | 0.28 | optimal | 0.79 | 7.00 |
| `ex9_2_8` | optimal | 0.00 | optimal | 0.01 | 100.00 |
| `fac1` | optimal | 0.56 | optimal | 0.65 | 2.00 |
| `filter` | optimal | 0.06 | time_limit | 60.96 | 4828.52 |
| `flay02h` | optimal | 0.80 | optimal | 1.09 | 1.00 |
| `flay02m` | optimal | 0.31 | optimal | 0.74 | 1.00 |
| `flay03m` | optimal | 0.68 | optimal | 1.01 | 1.12 |
| `flay04m` | optimal | 3.01 | optimal | 3.40 | 1.00 |
| `fuel` | optimal | 0.02 | optimal | 1.11 | 163.50 |
| `gbd` | optimal | 0.00 | optimal | 0.00 | 1.00 |
| `gear` | optimal | 0.13 | optimal | 0.02 | 0.00 |
| `gear2` | optimal | 7.67 | optimal | 0.11 | 0.00 |
| `gear3` | optimal | 0.18 | optimal | 0.02 | 0.00 |
| `gear4` | optimal | 0.48 | optimal | 1.95 | 8.39 |
| `genpooling_lee1` | optimal | 4.39 | time_limit | 60.00 | 26.17 |
| `genpooling_lee2` | optimal | 9.53 | time_limit | 60.00 | 9.31 |
| `ghg_1veh` | time_limit | 60.00 | time_limit | 60.00 | 0.01 |
| `gkocis` | optimal | 0.03 | optimal | 0.03 | 2.00 |
| `graphpart_2g-0044-1601` | optimal | 0.03 | optimal | 1.16 | 323.00 |
| `graphpart_2pm-0044-0044` | optimal | 0.09 | optimal | 3.53 | 331.00 |
| `graphpart_clique-20` | optimal | 2.02 | time_limit | 60.02 | 46.88 |
| `hadamard_4` | optimal | 0.28 | optimal | 0.66 | 6.20 |
| `hadamard_5` | optimal | 4.84 | time_limit | 60.00 | 15.89 |
| `haverly` | optimal | 0.02 | time_limit | 60.00 | 85425.00 |
| `himmel11` | optimal | 0.04 | optimal | 0.41 | 508.50 |
| `himmel16` | optimal | 4.81 | time_limit | 60.00 | 38.08 |
| `hmittelman` | optimal | 0.00 | optimal | 0.91 | 1.00 |
| `house` | optimal | 28.52 | time_limit | 60.00 | 2.19 |
| `hydro` | optimal | 0.07 | time_limit | 60.00 | 37795.00 |
| `jit1` | optimal | 0.06 | optimal | 3.14 | 626.67 |
| `kall_circles_c6a` | time_limit | 60.00 | time_limit | 60.00 | 0.57 |
| `kall_circles_c7a` | time_limit | 60.00 | time_limit | 60.00 | 0.50 |
| `kall_circlespolygons_c1p11` | optimal | 0.14 | optimal | 0.11 | 1.00 |
| `kall_circlesrectangles_c1r11` | optimal | 0.11 | optimal | 0.12 | 1.00 |
| `kall_congruentcircles_c31` | optimal | 0.46 | optimal | 0.48 | 0.53 |
| `kall_congruentcircles_c32` | optimal | 0.45 | optimal | 0.58 | 0.27 |
| `kall_congruentcircles_c41` | optimal | 0.45 | optimal | 0.60 | 0.48 |
| `kall_congruentcircles_c42` | optimal | 0.30 | optimal | 1.08 | 2.55 |
| `kall_congruentcircles_c51` | optimal | 4.78 | optimal | 4.80 | 0.69 |
| `kall_congruentcircles_c52` | optimal | 1.77 | optimal | 1.86 | 0.54 |
| `kall_congruentcircles_c62` | optimal | 5.40 | optimal | 5.07 | 0.61 |
| `kall_congruentcircles_c63` | optimal | 2.67 | optimal | 6.72 | 1.67 |
| `kall_congruentcircles_c71` | time_limit | 60.00 | time_limit | 60.00 | 0.50 |
| `kall_congruentcircles_c72` | optimal | 25.94 | optimal | 30.94 | 0.52 |
| `kall_diffcircles_10` | time_limit | 60.00 | time_limit | 60.00 | 0.57 |
| `kall_diffcircles_5a` | optimal | 2.33 | optimal | 4.00 | 0.45 |
| `kall_diffcircles_5b` | optimal | 5.02 | optimal | 6.98 | 0.41 |
| `kall_diffcircles_6` | optimal | 4.05 | time_limit | 60.00 | 4.33 |
| `kall_diffcircles_7` | optimal | 7.29 | time_limit | 60.00 | 4.66 |
| `kall_diffcircles_8` | optimal | 46.05 | time_limit | 60.00 | 0.58 |
| `kall_diffcircles_9` | time_limit | 60.00 | time_limit | 60.00 | 0.36 |
| `launch` | time_limit | 60.00 | time_limit | 60.00 | 0.23 |
| `least` | time_limit | 60.00 | time_limit | 60.00 | 103.10 |
| `lip` | optimal | 0.79 | optimal | 0.64 | 0.06 |
| `m3` | optimal | 0.49 | optimal | 0.28 | 0.53 |
| `mathopt1` | optimal | 0.06 | optimal | 0.01 | 0.04 |
| `mathopt2` | optimal | 0.00 | optimal | 0.00 | 1.00 |
| `mathopt3` | optimal | 0.00 | time_limit | 60.01 | 79917.00 |
| `mathopt4` | optimal | 0.23 | optimal | 0.00 | 0.00 |
| `mathopt5_1` | optimal | 0.02 | optimal | 0.10 | 1.00 |
| `mathopt5_2` | optimal | 0.04 | optimal | 0.04 | 1.00 |
| `mathopt5_3` | optimal | 0.10 | optimal | 0.15 | 0.72 |
| `mathopt5_4` | optimal | 4.90 | optimal | 0.02 | 0.00 |
| `mathopt5_5` | optimal | 0.15 | optimal | 0.13 | 0.03 |
| `mathopt5_6` | optimal | 0.17 | optimal | 0.07 | 1.00 |
| `mathopt5_7` | optimal | 0.20 | optimal | 0.45 | 0.80 |
| `mathopt5_8` | optimal | 0.23 | optimal | 0.35 | 1.21 |
| `mathopt6` | optimal | 0.67 | optimal | 0.69 | 0.18 |
| `meanvarx` | optimal | 0.18 | optimal | 2.98 | 132.60 |
| `nemhaus` | optimal | 0.00 | optimal | 0.00 | 100.00 |
| `nous1` | optimal | 6.14 | time_limit | 60.00 | 1.94 |
| `nous2` | optimal | 0.66 | time_limit | 60.00 | 132.76 |
| `nvs01` | optimal | 0.14 | optimal | 0.19 | 1.43 |
| `nvs02` | optimal | 0.03 | optimal | 0.09 | 18.00 |
| `nvs03` | optimal | 0.01 | optimal | 0.02 | 1.00 |
| `nvs04` | optimal | 0.17 | optimal | 0.01 | 0.03 |
| `nvs05` | time_limit | 60.00 | optimal | 19.94 | 0.15 |
| `nvs06` | optimal | 0.06 | optimal | 0.04 | 5.00 |
| `nvs07` | optimal | 0.01 | optimal | 0.01 | 1.00 |
| `nvs08` | optimal | 0.11 | optimal | 0.21 | 58.00 |
| `nvs09` | optimal | 0.12 | optimal | 1.60 | 2.84 |
| `nvs10` | optimal | 0.01 | optimal | 0.04 | 21.00 |
| `nvs11` | optimal | 0.06 | optimal | 0.09 | 16.50 |
| `nvs12` | optimal | 0.11 | optimal | 0.25 | 51.00 |
| `nvs13` | optimal | 0.69 | optimal | 2.67 | 76.93 |
| `nvs14` | optimal | 0.06 | optimal | 0.10 | 18.00 |
| `nvs15` | optimal | 0.01 | optimal | 0.03 | 4.00 |
| `nvs16` | optimal | 0.06 | optimal | 0.03 | 0.15 |
| `nvs17` | optimal | 2.43 | time_limit | 60.00 | 128.96 |
| `nvs18` | optimal | 0.79 | optimal | 15.24 | 80.83 |
| `nvs19` | optimal | 3.96 | time_limit | 60.00 | 66.86 |
| `nvs20` | optimal | 1.32 | optimal | 8.92 | 2.08 |
| `nvs21` | optimal | 0.04 | optimal | 0.10 | 0.95 |
| `nvs22` | optimal | 0.23 | optimal | 0.07 | 0.65 |
| `nvs23` | optimal | 8.89 | time_limit | 60.00 | 8.71 |
| `nvs24` | optimal | 12.72 | time_limit | 60.00 | 7.56 |
| `oaer` | optimal | 0.02 | optimal | 0.13 | 2.00 |
| `pointpack02` | optimal | 0.02 | optimal | 0.72 | 591.00 |
| `pointpack04` | optimal | 0.21 | optimal | 1.65 | 121.00 |
| `pointpack06` | optimal | 3.72 | optimal | 7.37 | 1.01 |
| `pointpack08` | optimal | 47.59 | time_limit | 60.00 | 0.46 |
| `pooling_adhya1pq` | optimal | 0.66 | optimal | 2.83 | 1.53 |
| `pooling_adhya1stp` | optimal | 2.50 | optimal | 6.39 | 0.90 |
| `pooling_adhya1tp` | optimal | 0.78 | optimal | 2.33 | 1.10 |
| `pooling_adhya2pq` | optimal | 0.50 | optimal | 1.60 | 0.83 |
| `pooling_adhya2stp` | optimal | 1.65 | optimal | 6.39 | 0.67 |
| `pooling_adhya2tp` | optimal | 0.59 | optimal | 1.93 | 0.83 |
| `pooling_adhya3pq` | optimal | 1.02 | optimal | 2.58 | 0.68 |
| `pooling_adhya3tp` | optimal | 1.17 | optimal | 2.17 | 0.39 |
| `pooling_adhya4pq` | optimal | 1.35 | optimal | 6.11 | 1.24 |
| `pooling_adhya4tp` | optimal | 1.26 | optimal | 5.28 | 1.12 |
| `pooling_bental4pq` | optimal | 0.11 | optimal | 0.27 | 1.67 |
| `pooling_bental4stp` | optimal | 0.09 | optimal | 0.75 | 17.00 |
| `pooling_bental4tp` | optimal | 0.09 | optimal | 0.28 | 1.00 |
| `pooling_foulds2pq` | optimal | 0.05 | optimal | 0.04 | 1.00 |
| `pooling_foulds2stp` | optimal | 1.61 | optimal | 1.90 | 0.50 |
| `pooling_foulds2tp` | optimal | 0.03 | optimal | 0.07 | 1.00 |
| `pooling_haverly1pq` | optimal | 0.01 | optimal | 0.03 | 1.00 |
| `pooling_haverly1stp` | optimal | 0.07 | optimal | 0.41 | 3.00 |
| `pooling_haverly1tp` | optimal | 0.01 | optimal | 0.04 | 1.00 |
| `pooling_haverly2pq` | optimal | 0.06 | optimal | 0.18 | 1.50 |
| `pooling_haverly2stp` | optimal | 0.15 | optimal | 0.46 | 3.50 |
| `pooling_haverly2tp` | optimal | 0.04 | optimal | 0.51 | 1.00 |
| `pooling_haverly3pq` | optimal | 0.01 | optimal | 0.02 | 1.00 |
| `pooling_haverly3stp` | optimal | 0.15 | optimal | 0.10 | 1.00 |
| `pooling_haverly3tp` | optimal | 0.04 | optimal | 0.01 | 0.50 |
| `pooling_rt2pq` | optimal | 0.65 | optimal | 1.25 | 0.38 |
| `pooling_rt2stp` | optimal | 1.25 | optimal | 2.97 | 1.52 |
| `pooling_rt2tp` | optimal | 0.53 | optimal | 1.32 | 1.51 |
| `portfol_buyin` | optimal | 0.64 | optimal | 2.79 | 16.25 |
| `portfol_card` | optimal | 0.52 | optimal | 20.75 | 201.75 |
| `powerflow0009r` | optimal | 40.06 | time_limit | 60.00 | 3.12 |
| `prob02` | optimal | 0.02 | optimal | 0.05 | 1.00 |
| `prob03` | optimal | 0.00 | optimal | 0.00 | 1.00 |
| `prob06` | optimal | 0.01 | optimal | 0.13 | 9.00 |
| `prob07` | time_limit | 60.00 | time_limit | 60.00 | 1.09 |
| `prob09` | optimal | 2.23 | optimal | 0.07 | 0.00 |
| `prob10` | optimal | 0.04 | optimal | 0.04 | 3.50 |
| `process` | optimal | 0.42 | optimal | 0.80 | 0.29 |
| `procsel` | optimal | 0.01 | optimal | 0.01 | 1.00 |
| `prolog` | optimal | 0.02 | time_limit | 60.00 | 228691.00 |
| `ramsey` | optimal | 0.40 | optimal | 1.61 | 1.00 |
| `rbrock` | optimal | 1.31 | optimal | 0.04 | 0.00 |
| `sep1` | optimal | 0.15 | optimal | 0.40 | 0.88 |
| `slay04m` | optimal | 0.70 | time_limit | 60.00 | 17878.50 |
| `smallinvDAXr1b010-011` | optimal | 0.98 | time_limit | 60.00 | 603.33 |
| `smallinvDAXr1b020-022` | optimal | 1.60 | time_limit | 60.00 | 123.57 |
| `smallinvDAXr1b050-055` | optimal | 4.15 | time_limit | 60.00 | 44.95 |
| `smallinvDAXr1b100-110` | optimal | 18.60 | time_limit | 60.01 | 11.97 |
| `smallinvDAXr1b150-165` | time_limit | 60.00 | time_limit | 60.00 | 4.62 |
| `smallinvDAXr1b200-220` | time_limit | 60.00 | time_limit | 60.00 | 5.70 |
| `smallinvDAXr2b010-011` | optimal | 0.67 | time_limit | 60.00 | 701.85 |
| `smallinvDAXr2b020-022` | optimal | 1.50 | time_limit | 60.00 | 219.14 |
| `smallinvDAXr2b050-055` | optimal | 15.51 | time_limit | 60.00 | 14.49 |
| `smallinvDAXr2b100-110` | optimal | 22.66 | time_limit | 60.00 | 7.37 |
| `smallinvDAXr2b150-165` | optimal | 54.87 | time_limit | 60.00 | 3.93 |
| `smallinvDAXr2b200-220` | optimal | 0.60 | time_limit | 60.00 | 423.68 |
| `smallinvDAXr3b010-011` | optimal | 0.63 | time_limit | 60.00 | 712.87 |
| `smallinvDAXr3b020-022` | optimal | 1.32 | time_limit | 60.00 | 160.17 |
| `smallinvDAXr3b050-055` | optimal | 7.09 | time_limit | 60.00 | 43.97 |
| `smallinvDAXr3b100-110` | optimal | 57.68 | time_limit | 60.00 | 2.93 |
| `smallinvDAXr3b150-165` | time_limit | 60.00 | time_limit | 60.00 | 2.89 |
| `smallinvDAXr3b200-220` | optimal | 0.62 | time_limit | 60.00 | 301.99 |
| `smallinvDAXr4b010-011` | optimal | 0.73 | time_limit | 60.00 | 682.75 |
| `smallinvDAXr4b020-022` | optimal | 1.75 | time_limit | 60.00 | 166.64 |
| `smallinvDAXr4b050-055` | optimal | 7.69 | time_limit | 60.00 | 34.04 |
| `smallinvDAXr4b100-110` | optimal | 22.53 | time_limit | 60.00 | 11.45 |
| `smallinvDAXr4b150-165` | optimal | 34.43 | time_limit | 60.00 | 6.02 |
| `smallinvDAXr4b200-220` | optimal | 0.77 | time_limit | 60.00 | 294.48 |
| `smallinvDAXr5b010-011` | optimal | 0.97 | time_limit | 60.01 | 553.75 |
| `smallinvDAXr5b020-022` | optimal | 2.15 | time_limit | 60.00 | 116.48 |
| `smallinvDAXr5b050-055` | optimal | 5.25 | time_limit | 60.01 | 68.48 |
| `smallinvDAXr5b100-110` | optimal | 30.98 | time_limit | 60.00 | 5.45 |
| `smallinvDAXr5b150-165` | optimal | 33.30 | time_limit | 60.00 | 10.35 |
| `smallinvDAXr5b200-220` | optimal | 0.87 | time_limit | 60.00 | 255.85 |
| `sporttournament06` | optimal | 0.04 | optimal | 0.47 | 610.00 |
| `sporttournament08` | optimal | 0.44 | optimal | 1.51 | 354.50 |
| `sporttournament10` | optimal | 0.39 | optimal | 18.23 | 176.14 |
| `sssd08-04` | optimal | 2.09 | time_limit | 60.00 | 1399.07 |
| `sssd08-04persp` | optimal | 11.80 | optimal | 46.00 | 3.31 |
| `st_bpaf1a` | optimal | 0.14 | optimal | 0.39 | 3.50 |
| `st_bpaf1b` | optimal | 0.01 | optimal | 0.01 | 1.00 |
| `st_bpk1` | optimal | 0.01 | optimal | 0.02 | 1.00 |
| `st_bpv1` | optimal | 0.00 | optimal | 0.00 | 100.00 |
| `st_bpv2` | optimal | 0.01 | optimal | 0.00 | 100.00 |
| `st_bsj2` | optimal | 0.11 | optimal | 0.13 | 1.27 |
| `st_bsj3` | optimal | 0.00 | optimal | 0.00 | 1.00 |
| `st_bsj4` | optimal | 0.05 | optimal | 0.13 | 1.00 |
| `st_e01` | optimal | 0.01 | optimal | 0.00 | 1.00 |
| `st_e02` | optimal | 0.00 | optimal | 0.03 | 1.00 |
| `st_e03` | optimal | 0.35 | time_limit | 60.00 | 287.41 |
| `st_e04` | optimal | 0.38 | optimal | 1.01 | 4.00 |
| `st_e05` | optimal | 0.19 | error | - | - |
| `st_e06` | optimal | 0.00 | optimal | 0.01 | 1.00 |
| `st_e07` | optimal | 0.03 | optimal | 0.21 | 10.50 |
| `st_e08` | optimal | 0.01 | optimal | 0.02 | 1.00 |
| `st_e09` | optimal | 0.01 | optimal | 0.22 | 45.00 |
| `st_e11` | optimal | 0.63 | optimal | 0.19 | 0.48 |
| `st_e12` | optimal | 0.00 | optimal | 0.00 | 1.00 |
| `st_e13` | optimal | 0.00 | optimal | 0.02 | 1.00 |
| `st_e14` | optimal | 0.05 | optimal | 0.10 | 2.00 |
| `st_e15` | optimal | 0.00 | optimal | 0.03 | 2.00 |
| `st_e16` | optimal | 0.09 | optimal | 1.21 | 6.62 |
| `st_e18` | optimal | 0.05 | optimal | 0.16 | 21.00 |
| `st_e19` | optimal | 0.10 | optimal | 0.35 | 1.51 |
| `st_e21` | optimal | 0.14 | optimal | 0.01 | 0.50 |
| `st_e22` | optimal | 0.01 | optimal | 0.02 | 1.00 |
| `st_e23` | optimal | 0.02 | optimal | 0.10 | 41.00 |
| `st_e24` | optimal | 0.04 | optimal | 0.12 | 1.00 |
| `st_e25` | optimal | 0.15 | optimal | 0.24 | 1.40 |
| `st_e26` | optimal | 0.01 | optimal | 0.02 | 1.00 |
| `st_e27` | optimal | 0.01 | optimal | 0.01 | 1.00 |
| `st_e28` | optimal | 0.02 | optimal | 0.37 | 508.50 |
| `st_e29` | optimal | 0.03 | optimal | 0.27 | 2.00 |
| `st_e30` | optimal | 0.25 | optimal | 0.76 | 0.93 |
| `st_e32` | optimal | 5.00 | optimal | 32.86 | 2.91 |
| `st_e33` | optimal | 0.01 | optimal | 0.21 | 23.00 |
| `st_e34` | optimal | 0.00 | optimal | 0.00 | 1.00 |
| `st_e35` | time_limit | 60.00 | time_limit | 60.00 | 0.33 |
| `st_e36` | optimal | 0.11 | optimal | 0.23 | 0.33 |
| `st_e38` | optimal | 0.04 | optimal | 0.08 | 1.50 |
| `st_e40` | optimal | 0.09 | optimal | 0.73 | 84.58 |
| `st_e41` | optimal | 0.60 | optimal | 0.05 | 0.05 |
| `st_e42` | optimal | 0.01 | optimal | 0.01 | 1.00 |
| `st_fp7a` | optimal | 0.28 | time_limit | 60.00 | 232.71 |
| `st_fp7b` | optimal | 0.38 | optimal | 12.94 | 55.96 |
| `st_fp7c` | optimal | 0.24 | time_limit | 60.00 | 231.04 |
| `st_fp7d` | optimal | 0.28 | optimal | 58.55 | 323.66 |
| `st_fp7e` | optimal | 1.11 | optimal | 0.95 | 0.48 |
| `st_fp8` | optimal | 0.23 | optimal | 0.46 | 3.66 |
| `st_glmp_fp1` | optimal | 0.01 | optimal | 0.02 | 1.00 |
| `st_glmp_fp2` | optimal | 0.01 | optimal | 0.21 | 13.00 |
| `st_glmp_fp3` | optimal | 0.02 | optimal | 0.31 | 11.00 |
| `st_glmp_kk90` | optimal | 0.01 | optimal | 0.01 | 1.00 |
| `st_glmp_kk92` | optimal | 0.01 | optimal | 0.18 | 11.00 |
| `st_glmp_kky` | optimal | 0.01 | optimal | 0.11 | 13.00 |
| `st_glmp_ss1` | optimal | 0.06 | optimal | 0.31 | 2.60 |
| `st_glmp_ss2` | optimal | 0.01 | optimal | 0.06 | 5.00 |
| `st_ht` | optimal | 0.04 | optimal | 0.20 | 1.00 |
| `st_iqpbk1` | optimal | 0.26 | optimal | 0.26 | 0.95 |
| `st_iqpbk2` | optimal | 0.23 | optimal | 0.18 | 0.95 |
| `st_jcbpaf2` | optimal | 0.10 | optimal | 0.15 | 0.23 |
| `st_m1` | optimal | 0.41 | optimal | 0.79 | 1.45 |
| `st_m2` | optimal | 0.24 | time_limit | 60.00 | 523.59 |
| `st_miqp1` | optimal | 0.00 | optimal | 0.00 | 100.00 |
| `st_miqp2` | optimal | 0.00 | optimal | 0.04 | 7.00 |
| `st_miqp3` | optimal | 0.01 | optimal | 0.00 | 1.00 |
| `st_miqp4` | optimal | 0.01 | optimal | 0.00 | 1.00 |
| `st_miqp5` | optimal | 0.01 | optimal | 0.69 | 145.00 |
| `st_pan1` | optimal | 0.05 | optimal | 0.12 | 0.64 |
| `st_ph1` | optimal | 0.03 | optimal | 0.03 | 1.00 |
| `st_ph10` | optimal | 0.00 | optimal | 0.00 | 1.00 |
| `st_ph11` | optimal | 0.05 | optimal | 0.15 | 1.00 |
| `st_ph12` | optimal | 0.04 | optimal | 0.12 | 1.00 |
| `st_ph13` | optimal | 0.05 | optimal | 0.17 | 1.00 |
| `st_ph14` | optimal | 0.01 | optimal | 0.01 | 1.00 |
| `st_ph15` | optimal | 0.06 | optimal | 0.10 | 1.67 |
| `st_ph2` | optimal | 0.03 | optimal | 0.01 | 1.00 |
| `st_ph20` | optimal | 0.05 | optimal | 0.19 | 2.50 |
| `st_ph3` | optimal | 0.03 | optimal | 0.01 | 1.00 |
| `st_phex` | optimal | 0.01 | optimal | 0.01 | 1.00 |
| `st_qpc-m0` | optimal | 0.02 | optimal | 0.06 | 1.00 |
| `st_qpc-m1` | optimal | 0.09 | optimal | 0.15 | 1.47 |
| `st_qpc-m3a` | optimal | 0.65 | optimal | 3.16 | 4.89 |
| `st_qpc-m3b` | optimal | 0.20 | optimal | 0.23 | 5.00 |
| `st_qpc-m3c` | optimal | 0.01 | optimal | 0.01 | 1.00 |
| `st_qpc-m4` | optimal | 0.01 | optimal | 0.01 | 1.00 |
| `st_qpk1` | optimal | 0.05 | optimal | 0.07 | 1.00 |
| `st_qpk2` | optimal | 0.13 | optimal | 0.25 | 2.04 |
| `st_qpk3` | optimal | 0.20 | optimal | 0.40 | 2.38 |
| `st_robot` | optimal | 0.03 | optimal | 0.02 | 1.00 |
| `st_rv1` | optimal | 0.20 | optimal | 0.76 | 0.82 |
| `st_rv2` | optimal | 0.25 | optimal | 0.59 | 3.87 |
| `st_rv3` | optimal | 0.54 | optimal | 1.02 | 1.71 |
| `st_rv7` | optimal | 1.22 | optimal | 2.92 | 3.30 |
| `st_rv8` | optimal | 1.27 | optimal | 2.30 | 0.38 |
| `st_rv9` | optimal | 1.69 | optimal | 2.89 | 0.88 |
| `st_test1` | optimal | 0.00 | optimal | 0.17 | 2.00 |
| `st_test2` | optimal | 0.00 | optimal | 0.00 | 100.00 |
| `st_test3` | optimal | 0.00 | optimal | 0.01 | 100.00 |
| `st_test4` | optimal | 0.00 | optimal | 0.00 | 1.00 |
| `st_test5` | optimal | 0.00 | optimal | 0.00 | 100.00 |
| `st_test6` | optimal | 0.00 | optimal | 0.00 | 100.00 |
| `st_test8` | optimal | 0.00 | optimal | 0.03 | 2.00 |
| `st_testgr1` | optimal | 0.10 | optimal | 0.58 | 13.81 |
| `st_testgr3` | optimal | 0.11 | optimal | 0.31 | 16.50 |
| `st_testph4` | optimal | 0.00 | optimal | 0.00 | 1.00 |
| `st_z` | optimal | 0.09 | optimal | 0.18 | 1.00 |
| `supplychain` | optimal | 0.27 | optimal | 0.83 | 3.11 |
| `syn05hfsg` | optimal | 0.06 | optimal | 0.13 | 1.67 |
| `syn05m` | optimal | 0.01 | optimal | 0.08 | 1.00 |
| `syn05m02m` | optimal | 0.13 | optimal | 0.08 | 0.50 |
| `syn10m` | optimal | 0.03 | optimal | 0.07 | 1.00 |
| `syn15m` | optimal | 0.07 | optimal | 0.19 | 2.00 |
| `synthes1` | optimal | 0.03 | optimal | 0.14 | 4.50 |
| `synthes2` | optimal | 0.10 | optimal | 0.27 | 1.00 |
| `synthes3` | optimal | 0.07 | optimal | 0.33 | 1.50 |
| `tanksize` | optimal | 2.48 | time_limit | 60.00 | 15.64 |
| `tln2` | optimal | 0.01 | optimal | 0.01 | 1.00 |
| `tln4` | optimal | 0.25 | optimal | 4.03 | 85.66 |
| `tln5` | optimal | 0.27 | time_limit | 60.00 | 1982.67 |
| `tln6` | optimal | 0.56 | time_limit | 60.00 | 138.46 |
| `tloss` | optimal | 0.06 | optimal | 0.14 | 62.00 |
| `tls2` | optimal | 0.55 | optimal | 0.75 | 0.75 |
| `tltr` | optimal | 0.03 | optimal | 0.09 | 2.00 |
| `trig` | optimal | 0.04 | optimal | 0.20 | 7.67 |
| `wallfix` | optimal | 0.06 | time_limit | 60.00 | 51745.60 |
| `wastewater02m1` | optimal | 0.12 | optimal | 0.31 | 1.14 |
| `wastewater02m2` | optimal | 0.43 | optimal | 0.67 | 1.20 |
| `wastewater04m1` | optimal | 0.19 | optimal | 0.67 | 1.15 |
| `wastewater04m2` | optimal | 0.24 | optimal | 0.79 | 0.96 |
| `wastewater05m1` | optimal | 3.92 | time_limit | 60.00 | 8.54 |
| `wastewater15m1` | optimal | 4.42 | time_limit | 60.00 | 4.44 |
| `windfac` | optimal | 0.06 | time_limit | 60.00 | 8156.75 |
