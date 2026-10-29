# Result layout

```Result layout and small I/O helpers.

Layout (all names are stable, so later stages can find earlier outputs):

    {drive_root}/                              e.g. MyDrive/GSoC_SuperRes/experiments or $SCRATCH/superres/experiments
      {dataset}/
        {date}_{version}_{tag}/                one experiment, e.g. 2026-10-06_v2_multiseed
          README_layout.md
          run_info.json                        start time, git commit, GPU, config of every session
          data_meta.json                       which events were cached (counts, scales, source files)
          models/
            vqvae_tokenizer/                   multi-scale VQ-VAE (one per experiment)
            var_transformer_{level}/           conditional VAR, one per LR level, e.g. var_transformer_pool2x2
            resnet18_tagger_{input}_seed{seed}/ qg quark/gluon tagger, e.g. resnet18_tagger_sr-pool2x2_seed42
            cnn_tagger_{input}_seed{seed}/     the same with tagger.arch=cnn
            cnn_regressor_{input}_seed{seed}/  calo incident-energy regressor
                config.json  history.csv  metrics.json  best.pt  last.pt  figures/
          evaluation/
            sr_{level}/                        physics observables, paired.csv, metrics.json, figures/
                figures/diagnostics/           the same with diagnostic methods added
          tuning/                              Optuna studies: {name}.db, {name}_trials.csv, {name}_best.json
          summary/                             {name}_report.md, tables (*.csv), figures/
                                               tagger files name the tagger, e.g. {name}_resnet18-tagger-2x3_pool2x2.csv

name    : {dataset}_{version}_{date}_{tag}, e.g. qg_v2_2026-10-06_multiseed. Every file name starts with it
          (figures: {name}_{model or level}_{what}.png), so a file copied elsewhere still says where it is from.
          '_' separates fields, '-' joins words inside a field.
input   : hr | lr-{level} | sr-{level} | uniform-{level} | srproj-{level}
Every figure is written as .png and .pdf.
```
