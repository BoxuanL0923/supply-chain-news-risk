# Q1 Scheme A: optimizer recovery and evidence identifiability

This is an engineering-only diagnostic of the cached smooth score, not a causal or financial validation.

## Plain verdict

**(i) Optimizer convergent with signal: FAIL under the pre-declared joint convergence-and-recovery criterion.** A failed recovery criterion does not by itself prove faulty optimization: weak curvature, confounding, noise, and the fixed stopping budget can all prevent parameter recovery.

**(ii) Real data identifiable: NO.** The signed-profile fit has 0 Hessian eigenvalues below 1e-6; all starts agree: False. At the midpoint, training correlation is -0.0669, validation correlation is 0.0653. Signed VP validation MSE is 1.698850 against zero-score 1.525929. These small development samples do not establish a reproducible, admissible predictive signal; they also cannot prove population signal is absent.

The real-data corner is allowed. Negative signed scales are diagnostics, not admissible prompt settings. Ridge regularization supplies prior curvature and cannot establish data identification.

## What was run and registered

Protocol: `D:\OptResearch_data\agent_q1\run_v2_identify\PROTOCOL_IDENTIFY.json`. Registered UTC: 2026-10-26T23:20:48.277597+00:00. Module SHA-256: `d1332624d5db854945c7c7f049aec4f57b470d707f307ad1beab3d7f0fa66510`. User-supplied original-run numbers were known in advance and recorded; no label file was read before registration.

The cached qwen3.8-27b evidence comprises 80 Aug–Sep training and 40 October validation cases, sealed before 2025-11-03. All runs use normalized unit-box parameters, fixed evidence budget eight, seed 20261026, the original five uniform [.1,.9] starts, step 0.5, 400 iterations, projected-gradient tolerance 1e-5, relative-loss tolerance 1e-7 and patience five. VP/MAP use constant and Robbins–Monro schedules. Synthetic fits use constant only. No validation-based selection, network requests or LLM calls were made.

The score/Jacobian are vectorized across cases. Repeated synthetic evidence is compressed into exact counts, sums of labels, and sums of squared labels; this retains the original MSE and gradient. The original `prompt_gd.descend` performs every update and stopping decision. For VP it receives six shape gradients and a zero dummy seventh gradient; the dummy coordinate is excluded from reported ranges and Hessians.

## D1: signed scale diagnostic

Here y means outcome minus anchor. Scale is s'y/s's without an intercept; a negative value is inadmissible. Pearson correlation is centered, whereas this scale uses uncentered inner products, so their signs need not agree in general.

| Fixed shape | Signed w | Train correlation | Valid correlation | Train signed MSE | Valid signed MSE |
|---|---:|---:|---:|---:|---:|
| DEFAULT | -5.14321 | -0.0583972 | 0.0882138 | 0.852157 | 1.70208 |
| midpoint | -5.41155 | -0.0669029 | 0.0652522 | 0.843019 | 1.70321 |

## Variable projection

All ten VP runs reached the 400-iteration cap. The reported Hessian describes the training-best returned iterate, not a certified stationary minimum. Its positive eigenvalues therefore do not override the failed start-agreement criterion.

Training-best shape: [0.240267, 1, 1, 0.351305, 0.095884, 0]. Signed scale: -5.52652 (negative). Train R² versus zero: 0.170458; conventional centered R²: -0.107312.

Final shape ranges over 5 starts × 2 schedules: [0.676145, 0, 1, 0.759025, 0.582461, 0.0698244]. All agree at range ≤0.05: False.

Finite-difference Hessian eigenvalues: [0.00200559, 0.00407019, 0.0129813, 0.0188173, 0.0335369, 0.126056]. Half-step check: [0.00200559, 0.00407019, 0.0129813, 0.0188173, 0.0335369, 0.126056]; largest Hessian entry difference 1.78604e-08. These are the symmetric Hessian of the smooth extension at boundary solutions, not a constrained uniqueness certificate; negative eigenvalues count as below cutoff.

| Prediction | Validation MSE | Gain vs zero |
|---|---:|---:|
| Zero score | 1.52593 | 0 |
| VP signed | 1.69885 | -0.172921 |
| VP nonnegative | 1.52593 | 0 |
| VP unit_box | 1.52593 | 0 |

`nonnegative` uses exactly max(0,w), as requested; `unit_box` additionally enforces the original upper bound one. All scales were fitted on train only.

## MAP: all registered ridge strengths

Agreement is empirical agreement across starts and schedules, not a proof of a globally unique minimum. All seven coordinates are penalized toward DEFAULT.

| rho / mean(y²) | rho | All agree | Max theta range | Train MSE | Valid MSE | Gain vs zero |
|---|---:|---|---:|---:|---:|---:|
| 0.001 | 0.000992506 | False | 0.55378 | 0.992506 | 1.52593 | 0 |
| 0.01 | 0.00992506 | False | 0.258873 | 0.992506 | 1.52593 | 0 |
| 0.1 | 0.0992506 | True | 0.00392434 | 1.00445 | 1.52458 | 0.00135273 |

| Parameter | DEFAULT | rho×0.001: estimate / signed movement | rho×0.01: estimate / signed movement | rho×0.1: estimate / signed movement |
|---|---:|---:|---:|---:|
| lambda2 | 0.5 | 0.43918 / -0.0608196 | 0.510115 / 0.0101147 | 0.496431 / -0.00356858 |
| lambda3 | 0.05 | 0.334561 / 0.284561 | 0.0647584 / 0.0147584 | 0.0449366 / -0.00506337 |
| THRESH_HUB | 0.8 | 0.681998 / -0.118002 | 0.801044 / 0.00104388 | 0.792704 / -0.00729615 |
| THRESH_RA | 0.5 | 0.458419 / -0.0415811 | 0.510064 / 0.0100637 | 0.500812 / 0.000812351 |
| THRESH_CLUST | 0.5 | 0.524638 / 0.0246382 | 0.501976 / 0.00197556 | 0.499315 / -0.000685266 |
| materiality | 0.5 | 0.330192 / -0.169808 | 0.504896 / 0.0048957 | 0.509088 / 0.00908766 |
| w | 0.5 | 0 / -0.5 | 0 / -0.5 | 0.215022 / -0.284978 |

All per-parameter ranges and absolute prior distances, all starts, stopping reasons, gradients and trajectories are in `identify.json`. The largest ridge yields agreement but moves each shape parameter less than 0.01 from DEFAULT. Its validation improvement of 0.00135273 (about 0.089%) is small and is not a significance or signal-detection result; all three registered strengths are reported without selecting one on validation.

## Planted-signal recovery

All 200 fits in the four required strong-signal cells exhausted 400 iterations; none met either convergence stop rule. Their corner rate was zero, yet agreement was also zero in every cell. Thus the recovery failure persists away from the real-data corner and with known signal.

Theta* is DEFAULT = [0.5, 0.05, 0.8, 0.5, 0.5, 0.5, 0.5]. For positive r, y=w*s(theta*)+epsilon, with independent Gaussian noise and sigma=std(w*s(theta*)) sqrt(r^-2−1), calibrated over the uniform empirical bank of all 120 cases. Thus r is population correlation in that empirical design, not an enforced finite-sample correlation. At n=80, evidence is sampled without replacement; larger training samples and all fresh validation samples use replacement. Each cell has ten seeded replications and five starts. SeedSequence([20261026,410,r_index,n_index,replication]) fixes all draws. The r=0 row is pure noise with sigma equal to planted-score standard deviation: finite independent Gaussian noise cannot make a nonzero fixed signal exactly uncorrelated. Its errors against DEFAULT are descriptive only; true shape recovery is undefined in the null.

MAE and GN eigenvalue counts use the minimum-training-loss start within each replication. Corner share uses all 50 runs. Validation gain is zero-score MSE minus fitted MSE on an independent synthetic draw of the same n; no model is chosen using it.

| r | n | Agreement share | Max parameter MAE | Mean flat GN count | Corner share | Mean valid gain | SD valid gain | Realized train corr | Pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 80 | 0.00 | 0.4748 | 5.70 | 0.10 | -2.76914e-06 | 1.05561e-05 | 0.035 | False |
| 0 | 200 | 0.00 | 0.4824 | 6.00 | 0.02 | -2.33705e-06 | 3.09268e-06 | -0.009 | False |
| 0 | 500 | 0.00 | 0.4788 | 6.00 | 0.02 | -4.65474e-06 | 8.1777e-06 | 0.017 | False |
| 0 | 2000 | 0.00 | 0.4820 | 6.00 | 0.00 | -2.34156e-06 | 2.03932e-06 | 0.007 | False |
| 0.1 | 80 | 0.00 | 0.5397 | 2.70 | 0.12 | 0.00124662 | 0.00239364 | 0.079 | False |
| 0.1 | 200 | 0.00 | 0.5619 | 1.80 | 0.00 | 0.000181914 | 0.00176691 | 0.099 | False |
| 0.1 | 500 | 0.00 | 0.4989 | 1.80 | 0.00 | 0.00202561 | 0.00118705 | 0.107 | False |
| 0.1 | 2000 | 0.00 | 0.5885 | 1.40 | 0.00 | 0.00112873 | 0.000346872 | 0.110 | False |
| 0.2 | 80 | 0.00 | 0.5570 | 2.10 | 0.02 | 0.00100811 | 0.000961262 | 0.158 | False |
| 0.2 | 200 | 0.00 | 0.5818 | 1.70 | 0.00 | 0.00141728 | 0.00111619 | 0.177 | False |
| 0.2 | 500 | 0.00 | 0.5865 | 1.40 | 0.00 | 0.00132386 | 0.000763952 | 0.192 | False |
| 0.2 | 2000 | 0.00 | 0.5328 | 1.30 | 0.00 | 0.00130516 | 0.000259222 | 0.198 | False |
| 0.4 | 80 | 0.00 | 0.5636 | 1.60 | 0.00 | 0.00102619 | 0.000660004 | 0.415 | False |
| 0.4 | 200 | 0.00 | 0.5386 | 1.50 | 0.00 | 0.00133743 | 0.000165643 | 0.382 | False |
| 0.4 | 500 | 0.00 | 0.5408 | 1.40 | 0.00 | 0.00125315 | 0.000207597 | 0.405 | False |
| 0.4 | 2000 | 0.00 | 0.5287 | 1.20 | 0.00 | 0.00129171 | 8.60216e-05 | 0.395 | False |

| r | n | lambda2 | lambda3 | THRESH_HUB | THRESH_RA | THRESH_CLUST | materiality | w |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 80 | 0.1172 | 0.4468 | 0.2012 | 0.0740 | 0.0481 | 0.2590 | 0.4748 |
| 0 | 200 | 0.0905 | 0.4245 | 0.1748 | 0.0620 | 0.0367 | 0.2531 | 0.4824 |
| 0 | 500 | 0.0902 | 0.4245 | 0.1748 | 0.0619 | 0.0367 | 0.2530 | 0.4788 |
| 0 | 2000 | 0.0902 | 0.4244 | 0.1749 | 0.0619 | 0.0367 | 0.2530 | 0.4820 |
| 0.1 | 80 | 0.3080 | 0.5397 | 0.1237 | 0.2221 | 0.1097 | 0.2125 | 0.2926 |
| 0.1 | 200 | 0.2237 | 0.5619 | 0.1192 | 0.1260 | 0.1551 | 0.2015 | 0.1500 |
| 0.1 | 500 | 0.2265 | 0.4989 | 0.1169 | 0.1824 | 0.0571 | 0.1909 | 0.0851 |
| 0.1 | 2000 | 0.3345 | 0.5885 | 0.1840 | 0.2503 | 0.0948 | 0.1993 | 0.0766 |
| 0.2 | 80 | 0.2506 | 0.5570 | 0.0844 | 0.2115 | 0.1039 | 0.1942 | 0.1741 |
| 0.2 | 200 | 0.3064 | 0.5818 | 0.1403 | 0.2379 | 0.0876 | 0.2169 | 0.1010 |
| 0.2 | 500 | 0.2745 | 0.5865 | 0.1387 | 0.2038 | 0.0874 | 0.1978 | 0.0510 |
| 0.2 | 2000 | 0.2152 | 0.5328 | 0.0539 | 0.1035 | 0.2415 | 0.1234 | 0.0473 |
| 0.4 | 80 | 0.3270 | 0.5636 | 0.1352 | 0.2345 | 0.1627 | 0.1838 | 0.0598 |
| 0.4 | 200 | 0.2593 | 0.5386 | 0.0982 | 0.1540 | 0.2150 | 0.1524 | 0.0583 |
| 0.4 | 500 | 0.2228 | 0.5408 | 0.0653 | 0.1397 | 0.1718 | 0.1405 | 0.0545 |
| 0.4 | 2000 | 0.2435 | 0.5287 | 0.0582 | 0.1374 | 0.2395 | 0.1159 | 0.0332 |

Minimum tested n meeting both ≥80% start agreement and ≤0.10 MAE for every parameter:

- r=0.0: not applicable (pure-noise null).
- r=0.1: not reached through n=2000.
- r=0.2: not reached through n=2000.
- r=0.4: not reached through n=2000.

Parameters exceeding the MAE threshold in at least one required strong-signal cell: lambda2, lambda3, THRESH_HUB, THRESH_RA, THRESH_CLUST, materiality.

Real data sit at n=80, |training correlation|=0.0669, between the r=0 and 0.1 grid rows; the actual sign is negative. This location is descriptive and is not a formal power estimate for the real-data distribution.

## Evidence geometry and weak parameters

Only the first eight evidence rows per case enter the score: 929 rows in total. Hop counts: {'0': 212, '1': 240, '2': 239, '3': 238}; nonzero signed-contribution counts: {'0': 136, '1': 82, '2': 60, '3': 53}.

| Parameter | DEFAULT GN diagonal | Jacobian column norm |
|---|---:|---:|
| lambda2 | 8.38433e-05 | 0.0709267 |
| lambda3 | 7.47079e-05 | 0.0669513 |
| THRESH_HUB | 9.89676e-05 | 0.0770588 |
| THRESH_RA | 1.6314e-06 | 0.00989364 |
| THRESH_CLUST | 1.33361e-06 | 0.0089452 |
| materiality | 0.000131796 | 0.0889254 |
| w | 0.0105587 | 0.79594 |

DEFAULT full-bank GN eigenvalues: [7.64778e-08, 3.91302e-07, 3.90269e-05, 5.57603e-05, 6.854e-05, 0.000150399, 0.0106368]. Weakest eigenvector in parameter order: [-0.00080918, 0.00288968, 0.0607351, 0.845259, 0.530882, -0.0017537, -0.000603523]. Diagonal curvature alone does not establish joint identification; correlated Jacobian columns can produce much weaker eigen-directions.

| Neighbor feature | Min | Median | Max | Std |
|---|---:|---:|---:|---:|
| degree_percentile | 65.1633 | 98.9114 | 99.9654 | 10.5401 |
| ra_percentile | 52.9208 | 97.9433 | 100 | 8.48047 |
| clustering | 0.00693988 | 0.0235625 | 0.121212 | 0.031299 |
| materiality | 0 | 0 | 0.95 | 0.195571 |

| Gate | Median sigmoid at DEFAULT | Fraction in [0.1,0.9] |
|---|---:|---:|
| THRESH_HUB | 0.709124 | 0.899582 |
| THRESH_RA | 0.908406 | 0.396095 |
| THRESH_CLUST | 0.0084561 | 0 |
| materiality | 0.00669285 | 0.105997 |

Individually nearly flat DEFAULT directions (GN diagonal <1e-6): none. In particular, the clustering gate is evaluated far above most observed clustering values when its threshold is 0.5, so its sigmoid derivative is small. The tables quantify the activation and limited graph variation; the full Jacobian cosine matrix is retained in JSON to expose confounding. Weak gate activation limits recovery even when labels are generated by this model; a large n formed by resampling cannot add new graph support.

Descriptive movement across the 200 required strong-signal fits:

| Parameter | Mean absolute movement from start |
|---|---:|
| lambda2 | 0.00950375 |
| lambda3 | 0.0114547 |
| THRESH_HUB | 0.0144417 |
| THRESH_RA | 0.00218023 |
| THRESH_CLUST | 0.00702938 |
| materiality | 0.0123392 |
| w | 0.202295 |

THRESH_RA and THRESH_CLUST are especially weak: they move only about 0.00218 and 0.00703 on average. Their DEFAULT Jacobian-column cosine is -0.8436, and the weakest GN eigenvector is dominated by RA (0.8453) and clustering (0.5309). Clustering values span only 0.00694 to 0.12121, far below the planted threshold 0.5; zero rows put that gate in the sigmoid transition region. RA percentiles concentrate near 98, also beyond the planted physical threshold 75. Only 53 of 238 H3 rows contribute nonzero signed evidence, and lambda3 moves just 0.01145 on average despite its large recovery error. These are weakly informative coordinates under this evidence and iteration budget, not exactly zero derivatives. Increasing n reduces noise but does not multiply the curvature of a mean-squared-error objective.

## Limitations

This tests a smooth engineering surrogate using previously cached LLM features, not fresh prompting or a structural economic model. The empirical bank has only 120 distinct cases; n=2000 reduces independent label noise conditional on resampled evidence, but is not 2000 independent real-world news environments. Training/validation synthetic draws can share evidence identities while their noise is independent. Gaussian synthetic noise and known correct model specification are favorable assumptions. Only ten replications estimate each cell's shares. The fixed 400-step, unscaled MSE gradient descent may stop before weak coordinates move appreciably. Failure of this exact protocol is not proof that no other optimizer or larger evidence design can recover parameters. Signed profiling permits negative or >1 scales solely for diagnosis; ridge can stabilize a prior-driven answer without creating evidence signal. No tested sample size beyond 2000 is inferred, and no interior real-data solution was forced.

## Possible errors and checks performed

- Vectorization: all 120 cached cases at 20 seeded random theta, max score discrepancy 5.55112e-17, max Jacobian discrepancy 5.55112e-17, both below 1e-12.
- All nine new unittests passed using `C:/OptResearch/.venv_dspy/Scripts/python.exe -m unittest examples.test_q1_identify -v` with `PYTHONPATH=source/code`, `OMP_NUM_THREADS=1`, and `Q1_SEAL_DATE=2025-11-03`. The suite checks boundary scores/Jacobians, finite-difference objective/profile/ridge gradients, duplicate-case sufficient statistics, scalar optimizer equivalence, and preregistration/validation guards.
- Hessian differencing: steps 1e-4 and 5e-5 agree to max entry difference 1.78604e-08; boundary smooth-extension interpretation is explicit.
- Cached model, case digest, feature IDs, host graph fields and feature bounds are checked; development origins and target sessions pass the original seal guards.
- Real labels are residualized once by subtracting anchor. Only original load_labels(train/valid) is used. The prediction freeze precedes validation access, and label access times are recorded.
- Source SHA-256 is registered before label access. The three protected source hashes are checked again after the experiment; the existing runner was not interrupted or modified.
- Synthetic seeds, realized correlations, all final fits, stopping reasons, eigenvalues, prior distances and fresh validation gains are retained for audit. Population-r calibration and the r=0 exception are registered explicitly.
- Native Windows peak-memory verification on an exact replay: 141426688 bytes (134.875 MiB), including figures and report, below 2 GiB. Both real and synthetic numerical results matched the first run exactly. The optional psutil meter was unavailable; `resource_verification.json` records the native measurement. Each experiment took about 46 seconds before plotting.

## Artifacts

- Full results: `D:\OptResearch_data\agent_q1\run_v2_identify\identify.json`
- Recovery figure: `D:\OptResearch_data\agent_q1\run_v2_identify\recovery_error.png`
- Corner/map figure: `D:\OptResearch_data\agent_q1\run_v2_identify\corner_rate.png`
- This report: `D:\OptResearch_data\external\supply-chain-news-risk\docs\Q1_IDENTIFY_REPORT.md`; copied to `C:\OptResearch\results\sc_risk\agent_q1\Q1_IDENTIFY_REPORT.md`.
