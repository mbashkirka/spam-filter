# Data directory

Raw and generated datasets are intentionally excluded from git and from milestone archives.

M1 downloads the **UCI SMS Spam Collection** (UCI dataset id 228) via `ucimlrepo` and creates:

```text
data/raw/sms_spam_collection.csv

data/processed/sms_spam_unique.csv

data/processed/train.csv

data/processed/validation.csv

data/processed/test.csv

data/processed/m1_manifest.json
```

Run from the repository root:

```bash
make m1-data
```

The source dataset is credited to Almeida & Hidalgo (2011), DOI `10.24432/C5CC84`, and is distributed by UCI under CC BY 4.0. Project code licensing does not replace the dataset license.
