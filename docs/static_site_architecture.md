# Static Site Architecture

The public website is designed as a static GitHub Pages site.

```text
Research server / local machine
  -> generate prediction and evaluation files
  -> export public JSON files into docs/data/
  -> push repository updates to GitHub
  -> GitHub Pages serves docs/
  -> user browser loads JSON and performs filtering, summaries, and charts
```

No public API is required for the current website. The browser only reads static
JSON files that have already been approved for public release.

## Data Layout

```text
docs/
├── index.html
├── static/
│   ├── app.js
│   └── styles.css
└── data/
    ├── manifest.json
    ├── predictions/
    │   ├── dates.json
    │   ├── latest.json
    │   └── YYYY-MM-DD.json
    ├── benchmark/
    │   └── leaderboard.json
    ├── backtesting/
    │   └── summary.json
    ├── analogues/
    │   └── examples.json
    ├── networks/
    │   └── latest.json
    ├── factors/
    │   └── catalog.json
    └── downloads/
        └── catalog.json
```

## Updating Public Prediction Data

After monthly prediction files are generated under:

```text
outputs/site_predictions/<country>/
```

export the public JSON files with:

```bash
python scripts/site/export_pages_data.py --country USA --prediction-root outputs/site_predictions
```

Then commit and push the updated `docs/data/` files.

## Security Boundary

Only public outputs should be exported to `docs/data/`.

Do not export:

```text
raw JKP characteristics
private research data
model checkpoints
server paths
credentials
```

The public site should expose only approved forecasts, benchmark summaries,
backtesting summaries, analogue summaries, and network summaries.
