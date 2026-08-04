# Toxiclassify project site

This repository serves the public project page at [toxiclassify.github.io](https://toxiclassify.github.io/). The site summarizes our 2025 machine learning course project and links to its code, notebook, and report.

The page uses plain HTML and CSS so GitHub Pages can publish it without a build step or third-party runtime.

The seven figures in `assets/figures` are generated from values saved in the executed course notebook. They cover label prevalence, co-occurrence, clipped comment length, baseline scores, model comparison, threshold errors, and the original cross-validation diagnostic. Recreate them with:

```powershell
python scripts\generate_figures.py
```

To preview it locally:

```powershell
python -m http.server 8000
```

Open `http://localhost:8000` in a browser. Changes pushed to `main` are published by GitHub Pages.
