# Baseline AUC Reproduction — Diagnostic Notes

**Status:** code-level diagnosis only. **No training was executed to produce this document** — this environment has no PyTorch install and no dataset-reading library (`pyarrow`) available, so `train_tagger()` could not actually be run here. See project issue #7 and the note at the end of this file on what running the reproduction requires.

## Context

Mentor review (Diptarko Choudhury, ~22:47 in the call) challenged this project's reported baseline tagger AUC (0.698, elsewhere reported as 0.669 depending on the run — see the README's headline table and [#4](https://github.com/rajveer43/CMS/issues/4)/PR #11), citing a comparable published jet-image classification result at approximately **0.8076** AUC. He asked for that published result to be reproduced independently before using any baseline AUC to contextualize tagging-efficiency numbers.

**The cited paper itself was not identified in the mentor call transcript or anywhere in this repo** — no arXiv link, title, or author was captured alongside the 0.8076 figure. I have not been able to confirm which specific paper is being referenced, so nothing below should be read as "reproducing paper X" — it is a diagnosis of *why this project's own tagger AUC is lower than a plausible published number on what looks like the same family of dataset*, grounded in this project's own code.

**Action needed from the user:** confirm (or share) the exact paper/slide source for the 0.8076 figure so a true apples-to-apples reproduction can be scoped. The dataset filenames present locally (`QCDToGGQQ_IMGjet_RH1all_jet0_run{0,1,2}_*.parquet`, CMS quark/gluon jet images) strongly suggest this is drawn from the ML4SCI CMS open-data quark/gluon jet-image line of work, which has several published CNN/ResNet baselines in the 0.7–0.81 AUC range depending on architecture and training-set size — but I have not verified a specific citation and won't assert one without confirming it.

## What the code actually does (verified by reading `classification_eval.py` and `multiscale_sr/data/`)

This is the part of the investigation that doesn't require running anything — the training-data volume the tagger sees is fully determined by the pipeline code, and it is small:

1. **Full available dataset:** 3 local parquet files, `n = 36,272 + 47,540 + 55,494 = 139,306` jets total (file names and row counts from `datasets/*.parquet` on disk).
2. **File-level split** (`split_files` in `multiscale_sr/data/normalization.py`, `val_ratio=0.33` default): with 3 files, `n_val_files = max(1, round(3*0.33)) = 1` — so **only the single smallest/last-sorted file is held out**; the other two are train-only and never seen by the tagger at all.
3. **Row-level split within that one held-out file** (`held_out_row_split`): splits by row-index parity — even rows → "val", odd rows → "test". So the tagger only ever draws from **half the rows of one file**.
4. **`classification_eval.py` loads the "test" split** (`get_dataloader(..., split="test", ...)`, line ~495) — i.e. odd rows of the one held-out file.
5. **`--max-samples` caps this further to 4000** (default, `classification_eval.py` line 88) via `collect_tagging_tensors`.
6. **The tagger's own 70/30 train/test split** (`_split_indices`, `test_frac=0.3` default) is drawn from that already-capped pool.

End state, matching the README's reported `n_train=2822, n_test=1210` (sums to ~4032, consistent with the `--max-samples 4000` cap): **the tagger trains on ~4,032 jets — about 2.9% of the 139,306 available — and only 2,822 of those are used for gradient updates.**

This alone is a strong, concrete, verified candidate explanation for at least part of the AUC gap. A CNN tagger (`JetTagger` in `multiscale_sr/tagger.py`: 3 conv layers, width 32, 15 epochs — deliberately "small... a consistent, quick-to-train probe," per its own docstring, not a competitive tagger) trained on ~2,800 examples is very plausibly undertrained relative to a published baseline trained on a much larger slice of the same ~139k-jet dataset family.

## Candidate causes, ranked by how directly they're supported by the code above

| # | Cause | Evidence | Confidence |
|---|---|---|---|
| 1 | **Training-set size.** Tagger sees ~2,822 samples vs. a paper likely trained on tens of thousands+. | Directly verified in code (§ above) — not a training pipeline design flaw, but a scope constraint that's correct for this project's purpose (measuring *tagging efficiency*, i.e. a relative HR/LR/SR comparison, not chasing SOTA AUC) but wrong to use for cross-paper AUC comparison without noting it. | High |
| 2 | **Tagger capacity/epochs.** `JetTagger` (3 conv layers, width 32) and 15 epochs is explicitly "compact... deliberately small" per its own docstring — likely far smaller than a paper's dedicated tagging architecture (e.g. ResNet-style, as used in several CMS open-data quark/gluon papers). | Directly verified (`tagger.py` docstring + architecture). `tagger_width`/`tagger_epochs` are already exposed as CLI flags (`tag_efficiency.py`, `classification_eval.py`), so this is testable without code changes. | High |
| 3 | **Different train/test split methodology.** This project's split is file-then-row-parity, purpose-built to guarantee zero leakage into SR-model training/val (see `held_out_row_split` docstring). A paper's published number likely uses a standard random/stratified split over the full dataset with no such held-out-for-a-different-model constraint. | Verified split logic; the *comparison target's* split is unknown (paper unidentified). | Medium — real difference, but can't quantify its effect without knowing the paper's methodology |
| 4 | **Different preprocessing/normalization.** This pipeline applies `log1p` + HR-referenced z-score normalization (`data/normalization.py`), computed on the *train* split only. A tagging paper optimized purely for classification (not also serving an SR pipeline) may preprocess differently (e.g. per-channel scaling tuned for the tagger, no log1p, different pixel binning/energy thresholding). | Verified this project's normalization; paper's preprocessing unknown. | Medium |
| 5 | **Genuine dataset difference.** If the paper uses a different production/run of the CMS open-data quark/gluon set (different jet pT range, different selection cuts, more or fewer input channels than this project's 3), some gap could be irreducible. | Cannot verify without the paper. | Low — most likely a minor contributor relative to #1/#2, but can't be ruled out |

**Read together, #1 and #2 are the most actionable and best-supported causes** — both are configuration choices in this project's own code (`--max-samples`, `--tagger-width`, `--tagger-epochs`), not evidence of an implementation bug. This reframes the mentor's question usefully: the 0.698 number is very likely correct *for the small-data regime this evaluation intentionally uses* (fast, consistent, fair across HR/LR/SR comparisons — the tagger's whole purpose per its docstring), but is not the number to quote as "our tagger's AUC" in a context implying it's competitive with a full-dataset published baseline.

## Reproduction recipe (not yet executed — requires an environment with PyTorch + pyarrow)

To actually test cause #1/#2 without touching the SR pipeline, `tag_efficiency.py`/`classification_eval.py` already expose everything needed via CLI flags — no code changes required for a first pass:

```bash
# Current default (small-data probe):
python classification_eval.py --checkpoint <ckpt> --data-dir ../datasets \
    --max-samples 4000 --tagger-width 32 --tagger-epochs 15

# Larger-data, larger-capacity probe (isolates cause #1/#2):
python classification_eval.py --checkpoint <ckpt> --data-dir ../datasets \
    --max-samples 40000 --tagger-width 64 --tagger-epochs 40
```

Note `--max-samples` is capped by how much data `get_dataloader(split="test", ...)` can actually provide (half of one held-out file, per §"What the code actually does" above) — to meaningfully exceed the current ~4k ceiling, `--val-ratio` would need to increase (more files held out) or a dedicated tagger-only data split outside the SR held-out set would need to be added, since reusing more of the *train* split here would leak SR-model training data into the tagger's training set (not necessarily wrong for a pure tagging-baseline check, but it would need to be a deliberate, documented choice, not an accident).

If AUC rises substantially toward 0.80 with more data/capacity, that confirms cause #1/#2 as the primary explanation and the fix is: **document the small-data tagger as an intentional fast relative-comparison tool, not a competitive-AUC claim**, and (optionally) add a separate, larger one-time "calibration" tagger run for absolute-AUC context. If AUC stays well below 0.80 even with substantially more data and capacity, that points at cause #3/#4/#5 instead and the paper's exact methodology needs to be pinned down.

## What changed in this PR

No training was run (see the environment limitation at the top). This PR:
- Adds this diagnostic document, tracing the actual sample-size and architecture constraints through the verified code path.
- Does **not** change README claims about AUC_HR/AUC_LR/AUC_SR numbers, since no new number was produced — updating those requires the reproduction recipe above to actually be run.
- Flags in the README that the current headline AUC numbers are measured on a small (~2.9% of available data) held-out slice, with a link to this document, so the caveat travels with the numbers rather than being lost.

## Next step

Whoever runs the reproduction recipe above (this needs a machine with PyTorch, `pyarrow`, and the dataset — e.g. the Colab environment prior runs in this project used) should fill in the results here and update the README's headline table + [`reports/multiscale_2026-07/REPORT.md`](reports/multiscale_2026-07/REPORT.md) with either the reproduced baseline or a confirmed "genuine setup difference" explanation, per the issue's acceptance criteria.
