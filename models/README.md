# Shared model weights

This folder holds every multi-hundred-MB model file used anywhere in the tree, deduplicated.

> **If you just cloned this repo from GitHub, this folder is empty.** The weights total ~10 GB and are excluded from version control (see `.gitignore`). Download them with the instructions below before running any pipeline that involves ChemEagle.

## Contents

| File / dir              | Size  | Used by                       |
|-------------------------|-------|-------------------------------|
| `ner.ckpt`              | 4.1 G | ChemEagle NER (chemiener)     |
| `biobert-large-cased/`  | 2.7 G | ChemEagle text agent          |
| `molnextr.pth`          | 1.1 G | ChemEagle molnextr            |
| `cre_models_v0.1/`      | 852 M | ChemEagle reaction extraction |
| `rxn.ckpt`              | 417 M | ChemEagle reaction model      |
| `moldet.ckpt`           | 383 M | ChemEagle molecule detection  |
| `corefdet.ckpt`         | 375 M | ChemEagle coreference         |
| `Tesseract-OCR/`        | 86 M  | ChemEagle OCR                 |

## How code finds these

`baselines/chemeagle/` and `approaches/chemeagle_gemini/` both contain **relative symlinks** pointing into this folder (e.g. `baselines/chemeagle/ner.ckpt → ../../models/ner.ckpt`). The original code paths therefore still work unchanged.

If you move or rename `5.Code_reorg/`, the relative symlinks survive because they don't encode an absolute prefix.

## How to obtain these weights

All eight come from the upstream ChemEagle release and are hosted on Hugging Face:

> **https://huggingface.co/CYF200127/ChemEAGLEModel/tree/main**

Download every file/folder listed there into `models/` so the layout matches the table above. With the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download CYF200127/ChemEAGLEModel \
    --local-dir <path-to-this-repo>/models \
    --local-dir-use-symlinks False
```

Or via `git` (the repo uses Git LFS):

```bash
git lfs install
git clone https://huggingface.co/CYF200127/ChemEAGLEModel /tmp/chemeagle_models
# Move the contents into <path-to-this-repo>/models/
```

After populating, run `bash tests/run_tests.sh` from the repo root — the `chemeagle_gemini/molnextr.pth resolves through symlink chain` check confirms the files are wired through correctly.

Sizes for an integrity sanity check (`du -sh models/*`):

| File / dir              | Expected size |
|-------------------------|---------------|
| `ner.ckpt`              | 4.1 GB |
| `biobert-large-cased/`  | 2.7 GB |
| `molnextr.pth`          | 1.1 GB |
| `cre_models_v0.1/`      | 852 MB |
| `rxn.ckpt`              | 417 MB |
| `moldet.ckpt`           | 383 MB |
| `corefdet.ckpt`         | 375 MB |
| `Tesseract-OCR/`        | 86 MB  |
