# Experiment results

This page reports aggregate results from the saved experiment summaries. The
fixed-configuration runs use leave-one-subject-out cross-validation (LOOCV)
with 36 subjects, seed 7, Adam, and frequency-only input. The analytical mean
correlation is 0.852393 across these runs. The reported correlations measure
spectral shape, not absolute power.

| Run | Hidden layers | Epochs | Learning rate | Mean held-out r | Mean gain vs. analytical | Subjects improved |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| basic_mlp_0 | [8] | 100 | 0.001 | 0.816314 | -0.036079 | 4/36 |
| basic_mlp_1 | [8] | 100 | 0.001 | 0.816956 | -0.035438 | 4/36 |
| basic_mlp_2 | [8] | 500 | 0.001 | 0.851140 | -0.001254 | 20/36 |
| basic_mlp_3 | [8] | 500 | 0.001 | 0.835533 | -0.016860 | 16/36 |
| basic_mlp_4 | [32] | 100 | 0.001 | 0.837671 | -0.014723 | 16/36 |
| basic_mlp_5 | [32] | 100 | 0.001 | 0.827987 | -0.024407 | 15/36 |
| basic_mlp_6 | [32] | 500 | 0.001 | 0.856241 | 0.003848 | 25/36 |
| basic_mlp_7 | [32] | 500 | 0.001 | 0.830316 | -0.022078 | 15/36 |
| basic_mlp_8 | [32, 32] | 100 | 0.001 | 0.837079 | -0.015314 | 16/36 |
| basic_mlp_9 | [32, 32] | 100 | 0.001 | 0.827650 | -0.024744 | 15/36 |
| basic_mlp_10 | [32, 32] | 500 | 0.001 | 0.851131 | -0.001263 | 20/36 |
| basic_mlp_11 | [32, 32] | 500 | 0.001 | 0.837679 | -0.014714 | 16/36 |
| basic_mlp_12 | [128, 128, 128, 128, 128] | 5000 | 0.001 | 0.860516 | 0.008123 | 27/36 |
| basic_mlp_13 | [128, 128, 128, 128, 128] | 5000 | 0.0003 | 0.862011 | 0.009617 | 31/36 |
| basic_mlp_14 | [32] | 5000 | 0.0003 | 0.857671 | 0.005278 | 26/36 |
| basic_mlp_15 | [32, 32] | 5000 | 0.0003 | 0.859865 | 0.007472 | 29/36 |
| basic_mlp_16 | [256, 256, 256, 256, 256] | 5000 | 0.0003 | 0.859295 | 0.006902 | 31/36 |
| basic_mlp_17 | [128, 128, 128, 128, 128, 128, 128] | 5000 | 0.0003 | 0.857271 | 0.004878 | 27/36 |
| basic_mlp_18 | [128, 128, 128] | 5000 | 0.0003 | 0.862559 | 0.010165 | 32/36 |
| basic_mlp_19 | [64, 64, 64, 64, 64] | 5000 | 0.0003 | 0.862428 | 0.010035 | 32/36 |

The [run-level ranking table](analysis/loocv-vs-analytical/ranking.csv) contains
aggregate metrics for runs 0–15. Run 18 has the highest mean held-out
correlation in this saved set. Model choices used this cohort, and each run
uses one seed, so these scores are exploratory and do not establish
independent performance or initialization robustness.

## Nested evaluation

The five-fold outer evaluation used five inner folds and six predeclared
candidates. Its mean outer held-out correlation was 0.862390 for the selected
MLP and 0.852393 for the analytical baseline, a raw-r difference of 0.009996.
The procedure selected run 12 for 22 outer subjects and run 13 for 14. The
paired Fisher-z test gave *t*(35) = 4.132 and unadjusted *p* = 0.000213.

The nested evaluation remains exploratory: prior model choices used this
cohort, outer training sets overlap, and the fitted SGM parameters come from
MEG. It does not establish structure-only prediction or absolute-power
calibration.

## Scope and provenance

This public summary contains cohort-level aggregates. It was prepared from the
saved run summaries; no training was rerun and no new result verification was
performed. Per-subject records and figures, model checkpoints, caches,
execution logs, machine-specific configuration paths, and archived launch
files remain outside this published subset.
