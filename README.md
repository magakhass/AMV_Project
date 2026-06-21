# Data

The model is trained on the **Real Estate Transactions** dataset published by the
Dubai Land Department (DLD) on the Dubai Pulse open-data portal. The data is
public but too large to commit, so it is git-ignored — follow the steps below to
obtain it.

## Obtaining the data

1. Go to the DLD **Real Estate Transactions** dataset on Dubai Pulse
   (`dubaipulse.gov.ae`, DLD organisation).
2. Download the CSV. The export is split into parts by size; a single part of
   ~1.3M Sales rows is more than enough for this project (the full set is not
   required).
3. Place the file at:

   ```
   data/raw/dld_transactions.csv
   ```

   If your filename differs, either rename it or update `paths.raw` in
   `config.yaml`.

The attribute dictionary used to design the schema is summarised in
`docs/data_dictionary.md`.

## Reproducing the processed data

From the repo root, run the pipeline stages in order:

```
python -m src.data         # load + Sales filter + typing      -> data/interim/
python -m src.preprocess   # outliers, bedrooms, nulls          -> data/interim/
python -m src.features     # date features + temporal split     -> data/processed/
python -m src.train        # baseline 3-model comparison        -> reports/, models/
python -m src.tune         # cross-validated tuning of HGB       -> models/
```

## Sample (optional)

To commit a tiny, runnable sample for graders, generate one from your raw file:

```python
import pandas as pd
(pd.read_csv("data/raw/dld_transactions.csv", nrows=2000)
   .to_csv("data/sample/transactions_sample.csv", index=False))
```

`data/sample/` is kept under version control (see `.gitignore`); the full
`data/raw`, `data/interim`, and `data/processed` folders are not.
