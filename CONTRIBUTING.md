# Contributing to CMS

Thank you for your interest in contributing to the CMS repository. This document explains how to get the project running locally, how to report issues, and how to prepare high-quality pull requests that are easy to review and merge.

Please read this guide before opening issues or PRs — it will speed up reviews and help maintainers respond faster.

---

## Table of contents

- Getting started (quick)
- Subproject quickstarts
  - MAEs / Hybrid_Transformer_Thanh_Nguyen
  - GNNs / GNN_for_momentum_estimation_Vishak_K_Bhat
  - E2E / E2E_DL_Reconstruction_Purva_Chaudhari
- Environment & dependencies
- Data and large files
- Making changes (workflow)
- Pull request checklist
- How to report bugs
- Code style and tests
- CI and smoke tests
- Communication & code of conduct
- Maintainers & contact

---

## Getting started (quick)

1. Fork the repository and clone your fork:
   - git clone https://github.com/<your-username>/CMS.git
   - cd CMS

2. Choose the subproject you want to work on (see Subproject quickstarts below) and follow its README to set up a working environment.

3. Create a branch for your work:
   - git checkout -b fix/<short-descriptive-name>

4. Make changes, run tests or smoke commands, and open a PR from your branch to `main` (or to the branch requested by the maintainers).

---

## Subproject quickstarts

This repo contains multiple independent subprojects. Follow the README in each subfolder for the canonical instructions; the snippets below are quick pointers.

### MAEs / Hybrid_Transformer_Thanh_Nguyen

- Path: `MAEs/Hybrid_Transformer_Thanh_Nguyen`
- Recommended: create an isolated Python environment (conda or venv).
- Install lightweight dependencies first; install PyTorch separately following https://pytorch.org/get-started/locally/ (choose correct CUDA/CPU wheel).
- Example:
  - python -m venv venv
  - source venv/bin/activate (Windows: venv\Scripts\activate)
  - python -m pip install --upgrade pip
  - pip install -r requirements.txt (or use `requirements-py310.txt` if provided)
  - Install torch/torchvision using the official PyTorch command for your CUDA.

Refer to `MAEs/Hybrid_Transformer_Thanh_Nguyen/README.md` for full instructions and available CLI scripts.

### GNNs / GNN_for_momentum_estimation_Vishak_K_Bhat

- Path: `GNNs/GNN_for_momentum_estimation_Vishak_K_Bhat`
- Uses conda in examples; create `conda create -n gsoc python=3.10` then `pip install -r requirements.txt`.
- Run training scripts like `python main.py` or `python train.py` with appropriate flags. See the subproject README for example commands.

### E2E / E2E_DL_Reconstruction_Purva_Chaudhari

- Path: `E2E/E2E_DL_Reconstruction_Purva_Chaudhari`
- Requires CMSSW (CMS software), `scram`, `cmsenv`, and ROOT. This is not a standard pip package.
- Typical steps:
  - scram p CMSSW\_<version>
  - cd CMSSW\_<version>/src
  - cmsenv
  - git clone the RecoE2E repo/branch
  - scram b -j <N>
- Use `cmsRun` with the example cfg file for inference. See the subproject README for details.

---

## Environment & dependencies

- Use isolated environments (conda or venv). On Windows, conda is often easier for binary packages.
- Python version: check the subproject README. If not specified, prefer Python 3.10 for wide compatibility.
- PyTorch: install using the official installer for correct CUDA support; do not rely on the generic `pip install torch` when GPU support is required.
- If a `requirements-py310.txt` or `environment-py310.yml` is provided, use those for reproducible installs.

Commands:

- Upgrade pip:
  - python -m pip install --upgrade pip
- Install without PyTorch (then install torch separately if needed):
  - pip install -r requirements-no-torch.txt
  - pip install <pytorch command from pytorch.org>

---

## Data and large files

- The repository references very large datasets (tens of GB). Do not commit large datasets to the repo.
- Use provided download scripts (e.g., `MAEs/.../scripts/download.py`) or host large files externally (Zenodo, S3, etc.).
- For pull requests, include or generate a small sample dataset (MBs) that exercises the code path — call it `data/sample/` — so reviewers can run smoke tests quickly.

If you must add large binaries, store them in an external host or use Git LFS (and coordinate with maintainers before adding).

---

## Making changes (workflow)

1. Fork the repo.
2. Create a branch with a descriptive name:
   - feature: `feature/<short-description>`
   - bugfix: `fix/<short-description>`
   - docs: `docs/<short-description>`
3. Make small, focused commits. Use meaningful commit messages (imperative present tense, short subject line, optional body).
   - Example: `Add requirements-py310 and README note for Python 3.10 users`
4. Run linters/tests locally before pushing.
5. Push the branch and open a PR.

---

## Pull request checklist

Before opening a PR, ensure:

- [ ] The PR targets the correct branch (`main`) unless maintainers specify otherwise.
- [ ] The PR has a descriptive title and an informative description.
- [ ] Changes are small and focused. Large changes may be split into multiple PRs.
- [ ] All tests or smoke checks pass locally.
- [ ] You included instructions to reproduce the change and, if applicable, a small sample dataset or a smoke command that reviewers can run.
- [ ] You did not commit large files or secrets. Use Git LFS for large assets.
- [ ] Documentation (README, subproject docs) updated when relevant.
- [ ] CI configuration or dependency changes are explained in the PR description.

Suggested PR template content (short):

- What and why: one or two sentences describing the change.
- How to test: exact commands to reproduce.
- Notes: compatibility or migration notes, if any.

---

## How to report bugs

When opening an issue, please include:

- Title: short summary
- Reproduction steps:
  1. Commands you ran
  2. Files/config used
- Expected behavior vs actual behavior
- Error messages / full stack trace (or a link to a Gist)
- Environment details:
  - OS and version
  - Python version
  - CUDA/cuDNN version (if relevant)
  - pip list or `pip freeze` output (trim to relevant packages)
- Subproject and git commit or tag (e.g., `MAEs/Hybrid_Transformer_Thanh_Nguyen`, commit `abcdef`)

A well-formed issue accelerates debugging and fixes.

---

## Code style and tests

- Python: follow PEP8. The maintainers recommend using:
  - black (formatting)
  - isort (import sorting)
  - flake8 (style checks) — optional
- Add unit tests where possible. For ML code, include small deterministic tests or smoke tests that run on tiny inputs.
- Type hints are encouraged where helpful.
- For linters and formatters, provide command examples in PRs:
  - python -m black .
  - python -m isort .

---

## CI and smoke tests

- If you add or change dependencies, update CI config (if present) or document changes in the PR.
- For heavy deps like PyTorch, CI should skip GPU installs or use a lightweight path for smoke checks.
- Add a minimal smoke test that can be run in CI on CPU, with small sample data.

Example smoke command (replace with real path inside each subproject):

- python -m scripts.evaluate_LorentzParT --help
- python GNNs/GNN_for_momentum_estimation_Vishak_K_Bhat/Scripts/main.py --help

---

## Communication & code of conduct

- Be respectful and constructive in discussions.
- If you are unsure about scope (breaking changes, major refactors, or large dataset additions), open an issue first and discuss your plan.
- If the project does not yet include a CODE_OF_CONDUCT, please follow standard open-source community norms.

---

## Maintainers & contact

- When opening PRs, request reviewers who have previously committed to the corresponding subproject (e.g., authors of the MAEs, GNNs, E2E subfolders).
- If you are unsure whom to tag as reviewer, add the `@Ankitaghavate` maintainers or open an issue to request guidance.

---

Thank you for helping improve the project — your contributions make the codebase stronger and more reproducible.

If you'd like, I can:

- open a PR that adds this `CONTRIBUTING.md` file to the repository, or
- generate an ISSUE/PR template and a `CODE_OF_CONDUCT.md` to go with it.
