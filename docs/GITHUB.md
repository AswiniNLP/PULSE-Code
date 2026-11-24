# Prepare the GitHub repository

Use this `code/` directory as the repository root. It contains the project README, license, dependency manifest, examples, tests, and CI workflow. The sibling archive is not part of this repository.

Run the checks before uploading:

```bash
python -m pip install -e '.[dev]'
python -m ruff check .
python -m unittest discover -s tests -v
python scripts/smoke_pipeline.py --output-dir artifacts/pre-upload
```

The smoke output directory must be empty or new. Review the validation report and preserve the distinction between tiny-model connectivity checks and full experimental validation.

If the folder is not already a Git repository:

```bash
git init -b main
```

Review the upload contents before committing:

```bash
git status --short --untracked-files=all
git add .
git diff --cached --stat
git diff --cached
git commit -m "Prepare PULSE training, inference, evaluation, and documentation"
```

Create an empty GitHub repository, then substitute your repository URL:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

`.gitignore` excludes real data, model weights, checkpoints, run outputs, caches, `.env` files, common key files, and virtual environments. It does not remove previously tracked files; inspect the staged diff even when ignore rules are present. Only the synthetic examples are intended to accompany the source.

The repository uses the MIT license selected by its owner. If publishing a paper or dataset later, add the actual citation and dataset provenance; this preparation does not invent either. Update the validation report after a real GPU and full-resource run.
