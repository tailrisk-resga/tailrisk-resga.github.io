# Public Site Export Utilities

These scripts build compact JSON and metadata files for the static dashboard in
`docs/`. They are intentionally separate from the paper reproduction pipeline in
`scripts/01_prepare_data.py` through `scripts/05_make_tables.py`.

The expected workflow is:

```bash
python scripts/site/export_resga_public_predictions.py --github-pages-root docs/data/predictions
python scripts/site/enrich_pages_stock_metadata.py --usa-id-csv /path/to/USA_id.csv
python scripts/site/export_public_backtests.py
python scripts/site/export_public_ebacktesting.py
python scripts/site/export_public_group_importance.py --github-pages-root docs/data/group_importance
```

Only files approved for public release should be written under `docs/data/`.
