# qg super-resolution summary: 2026-10-13_v2_multiseed

generated 2026-10-14 07:32:26

SR means the model output exactly as generated. Columns such as 'Uniform upsample', 'SR + energy constraint' and 'Tokenizer only' are diagnostics.

## Downstream ResNet-18 tagger on the test set. 'boot_err' is the uncertainty from resampling the test set; 'spread over seeds' needs at least two seeds.

|    | method   | level   |   n seeds |   auc (mean) |   auc (spread over seeds) |   accuracy (mean) |   accuracy (spread over seeds) |   rej_at_eff30 (mean) |   rej_at_eff30 (spread over seeds) |   rej_at_eff50 (mean) |   rej_at_eff50 (spread over seeds) |   rej_at_eff70 (mean) |   rej_at_eff70 (spread over seeds) |   auc_boot_err (mean) |   auc_boot_err (spread over seeds) |   rej_at_eff50_boot_err (mean) |   rej_at_eff50_boot_err (spread over seeds) |
|---:|:---------|:--------|----------:|-------------:|--------------------------:|------------------:|-------------------------------:|----------------------:|-----------------------------------:|----------------------:|-----------------------------------:|----------------------:|-----------------------------------:|----------------------:|-----------------------------------:|-------------------------------:|--------------------------------------------:|
|  0 | hr       | native  |         3 |       0.8086 |                    0.002  |            0.7343 |                         0.0012 |               23.1571 |                             1.397  |                9.4711 |                             0.3407 |                4.3533 |                             0.1482 |                0.0067 |                             0.0003 |                         0.7056 |                                      0.0616 |
|  1 | lr       | pool2x2 |         3 |       0.8075 |                    0.0019 |            0.7325 |                         0.0035 |               23.4335 |                             1.2995 |                9.7107 |                             0.3361 |                4.2769 |                             0.0828 |                0.0063 |                             0.0001 |                         0.6987 |                                      0.0591 |
|  2 | sr       | pool2x2 |         3 |       0.7871 |                    0.0022 |            0.7135 |                         0.0087 |               20.4254 |                             1.1433 |                7.8409 |                             0.2057 |                3.8743 |                             0.0626 |                0.0065 |                             0.0002 |                         0.5878 |                                      0.0022 |

## pool2x2: ResNet-18 tagger ROC AUC, A = trained on HR only and applied to each input (LR stretched back to the HR grid), B = trained and tested on each input

|                        | A: trained on HR          | B: trained on each input   |
|:-----------------------|:--------------------------|:---------------------------|
| HR                     | 0.8086 ± 0.0020 (3 seeds) | 0.8086 ± 0.0020 (3 seeds)  |
| LR                     | 0.8003 ± 0.0039 (3 seeds) | 0.8075 ± 0.0019 (3 seeds)  |
| SR                     | 0.7834 ± 0.0020 (3 seeds) | 0.7871 ± 0.0022 (3 seeds)  |
| SR + energy constraint | 0.8057 ± 0.0021 (3 seeds) | –                          |

## pool2x2: mean of each quantity for the truth, the coarse input and the super-resolved output (other columns are diagnostics)

| observable    |   HR (truth) | LR (input)   |   Super-resolved (VAR) |   Tokenizer only (HR→tokens→HR, with LR) |   SR, sampled tokens |   Uniform upsample |   SR + energy constraint |   SR distance to truth (W1/σ) |
|:--------------|-------------:|:-------------|-----------------------:|-----------------------------------------:|---------------------:|-------------------:|-------------------------:|------------------------------:|
| dR_std        |       0.3533 | 0.3531       |                 0.3719 |                                   0.3765 |               0.3736 |             0.3529 |                   0.3531 |                        0.4063 |
| girth         |       0.3336 | 0.3341       |                 0.3731 |                                   0.3942 |               0.3771 |             0.3345 |                   0.334  |                        0.3992 |
| jet_mass      |     121.434  | 121.4736     |                80.671  |                                  86.7174 |              80.8364 |           121.514  |                 121.461  |                        1.074  |
| jet_pt        |     234.496  | 234.4886     |               145.103  |                                 149.07   |             144.183  |           234.48   |                 234.491  |                        1.1208 |
| n_hits        |     679.448  | –            |               606.087  |                                 619.959  |             608.125  |          1242.19   |                 866.196  |                        0.3691 |
| n_hits_ecal   |     241.595  | –            |               190.966  |                                 199.069  |             193.78   |           516.936  |                 352.892  |                        0.4348 |
| n_hits_hcal   |     421.931  | –            |               419.306  |                                 420.128  |             419.424  |           559.523  |                 422.5    |                        0.0332 |
| n_hits_tracks |      78.3918 | –            |                40.3045 |                                  46.092  |              40.0677 |           308.626  |                 194.519  |                        1.3154 |
| ptD           |       0.1825 | –            |                 0.1385 |                                   0.1337 |               0.1361 |             0.1111 |                   0.1317 |                        0.6454 |
| sum_all       |     249.393  | 249.3928     |               155.69   |                                 160.835  |             154.86   |           249.393  |                 249.393  |                        1.1407 |
| sum_ecal      |      78.9779 | 78.9779      |                70.7552 |                                  70.5044 |              69.6122 |            78.9779 |                  78.9779 |                        0.2093 |
| sum_hcal      |      47.3665 | 47.3665      |                48.5422 |                                  48.2851 |              48.6619 |            47.3665 |                  47.3665 |                        0.0392 |
| sum_tracks    |     123.048  | 123.0484     |                36.3925 |                                  42.0452 |              36.5858 |           123.048  |                 123.048  |                        1.3593 |
| tau21         |       0.8338 | 0.8448       |                 0.8391 |                                   0.8358 |               0.8382 |             0.7346 |                   0.8211 |                        0.0621 |

## pool2x2: event by event against HR. bias = mean(X - HR) / mean(HR), resolution = std(X - HR) / mean(HR), pearson_r = correlation

| observable    | ('bias', 'lr')   |   ('bias', 'sr') | ('pearson_r', 'lr')   |   ('pearson_r', 'sr') | ('resolution', 'lr')   |   ('resolution', 'sr') |
|:--------------|:-----------------|-----------------:|:----------------------|----------------------:|:-----------------------|-----------------------:|
| dR_std        | -0.0005          |           0.0527 | 0.9998                |                0.909  | 0.0023                 |                 0.0541 |
| girth         | 0.0014           |           0.1184 | 0.9999                |                0.9047 | 0.0035                 |                 0.1383 |
| jet_mass      | 0.0003           |          -0.3357 | 1.0                   |                0.9404 | 0.0013                 |                 0.1483 |
| jet_pt        | -0.0             |          -0.3812 | 1.0                   |                0.9328 | 0.0002                 |                 0.1251 |
| n_hits        | –                |          -0.108  | –                     |                0.9829 | –                      |                 0.0628 |
| n_hits_ecal   | –                |          -0.2096 | –                     |                0.9761 | –                      |                 0.1428 |
| n_hits_hcal   | –                |          -0.0062 | –                     |                0.9976 | –                      |                 0.0256 |
| n_hits_tracks | –                |          -0.4859 | –                     |                0.8357 | –                      |                 0.2505 |
| ptD           | –                |          -0.2411 | –                     |                0.6988 | –                      |                 0.285  |
| sum_all       | -0.0             |          -0.3757 | 1.0                   |                0.9312 | 0.0                    |                 0.1234 |
| sum_ecal      | -0.0             |          -0.1041 | 1.0                   |                0.9933 | 0.0                    |                 0.0778 |
| sum_hcal      | 0.0              |           0.0248 | 1.0                   |                0.9988 | 0.0                    |                 0.0333 |
| sum_tracks    | 0.0              |          -0.7042 | 1.0                   |                0.8854 | 0.0                    |                 0.2408 |
| tau21         | 0.0132           |           0.0064 | 0.775                 |                0.5445 | 0.1077                 |                 0.1435 |

## Distance to the HR distribution, W1 / sigma_HR (0 = identical)

| observable        | ('pool2x2', 'lr')   |   ('pool2x2', 'sr') |   ('pool2x2', 'sr-sample') |   ('pool2x2', 'srproj') |   ('pool2x2', 'uniform') |   ('pool2x2', 'vqrec') |
|:------------------|:--------------------|--------------------:|---------------------------:|------------------------:|-------------------------:|-----------------------:|
| dR_std            | 0.0047              |              0.4063 |                     0.4452 |                  0.0047 |                   0.008  |                 0.5071 |
| girth             | 0.0052              |              0.3992 |                     0.4394 |                  0.0045 |                   0.0099 |                 0.6129 |
| jet_mass          | 0.0017              |              1.074  |                     1.0696 |                  0.0017 |                   0.0024 |                 0.9147 |
| jet_pt            | 0.0004              |              1.1208 |                     1.1323 |                  0.0004 |                   0.0004 |                 1.071  |
| log10_cell_energy | –                   |              0.0722 |                     0.0716 |                  0.0708 |                   0.2386 |                 0.076  |
| n_hits            | –                   |              0.3691 |                     0.3589 |                  0.9397 |                   2.8316 |                 0.2993 |
| n_hits_ecal       | –                   |              0.4348 |                     0.4106 |                  0.9558 |                   2.3645 |                 0.3652 |
| n_hits_hcal       | –                   |              0.0332 |                     0.0343 |                  0.0262 |                   0.8895 |                 0.0317 |
| n_hits_tracks     | –                   |              1.3154 |                     1.3236 |                  4.0107 |                   7.9515 |                 1.1155 |
| ptD               | –                   |              0.6454 |                     0.6804 |                  0.7448 |                   1.0477 |                 0.7165 |
| sum_all           | 0.0                 |              1.1407 |                     1.1508 |                  0      |                   0      |                 1.078  |
| sum_ecal          | 0.0                 |              0.2093 |                     0.2384 |                  0      |                   0      |                 0.2157 |
| sum_hcal          | 0.0                 |              0.0392 |                     0.043  |                  0      |                   0      |                 0.0308 |
| sum_tracks        | 0.0                 |              1.3593 |                     1.3563 |                  0      |                   0      |                 1.2707 |
| tau21             | 0.0881              |              0.0621 |                     0.0784 |                  0.1119 |                   0.7613 |                 0.1049 |

## Two-sample test AUC (0.5 = the small CNN cannot tell it from HR) and LR closure (sum|pool(SR) - LR| / sum LR; 0 = agrees with the coarse measurement)

|    | level   | method    | c2st_auc   | c2st_auc_boot_err   |   median_lr_closure |   mean_lr_closure |
|---:|:--------|:----------|:-----------|:--------------------|--------------------:|------------------:|
|  0 | pool2x2 | sr        | 0.9987     | 0.0004              |              0.4811 |            0.4727 |
|  1 | pool2x2 | sr-sample | –          | –                   |              0.4859 |            0.4788 |
|  2 | pool2x2 | srproj    | –          | –                   |              0      |            0      |
|  3 | pool2x2 | uniform   | –          | –                   |              0      |            0      |
|  4 | pool2x2 | vqrec     | –          | –                   |              0.4519 |            0.4455 |
