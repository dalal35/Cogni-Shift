# Input schema and provenance

Clinical datasets are not included in the current repository tree; earlier commits retain historical inputs as described below. A fully artificial example is built into `python -m cognishift --synthetic`; it needs no external data or credentials.

## Source and authorized access

The project’s original `poe.csv` and `emar.csv` inputs come from **MIMIC-IV**, hosted by PhysioNet. MIMIC-IV is restricted clinical research data, not an unrestricted public download. Researchers must obtain their own authorization: become a credentialed PhysioNet user, complete the required human-subjects research training, and sign the project’s Data Use Agreement. Follow the [official MIMIC-IV access requirements](https://physionet.org/content/mimiciv/3.1/) for current instructions. Access under that agreement does not authorize redistributing these files through GitHub.

The local input inventory is:

| File | Source and current role |
| --- | --- |
| `poe.csv` | Restricted MIMIC-IV provider-order extract; removed from the working directory and public release; obtain separately for the real evaluation |
| `emar.csv` | Restricted MIMIC-IV medication-administration extract; removed from the working directory and public release; not used by the current evaluator |
| `decision_fatigue_dataset.csv` | Redistributable CC0 synthetic behavioral dataset from Kaggle; not part of MIMIC-IV and not used by the current evaluator |

Original clinical copies and historical work remain in earlier commits and a private, Git-ignored local archive. Both removed working CSVs were verified byte-for-byte against that backup before removal. The exact MIMIC-IV release and original sampling/extraction query are not recorded, so another authorized extract will not necessarily reproduce the published sample or metrics.

After receiving access, obtain the POE table or an authorized extract through PhysioNet’s approved access route. Decompress it locally if necessary and place the CSV at `data/poe.csv`, or pass its location with `--input`. It must contain the four columns below; no EMAR or behavioral input is required. Do not place credentials, raw record previews, or extracts in issues, cloud tools, or commits. Keep local inputs and derived provider-level outputs within your approved research environment.

## Authorized evaluation input

Supply a CSV with these exact column names. Extra columns are ignored:

| Column | Meaning and handling |
| --- | --- |
| `order_provider_id` | Read as text to preserve distinct IDs such as `001` and `1`; missing or blank IDs are dropped |
| `ordertime` | Parseable timestamp in one consistent timezone convention; invalid values are dropped |
| `transaction_type` | `Change` and `D/C` count as modifications after stripping/uppercasing; other or missing values do not |
| `order_type` | Order category; missing values become `Unknown` |

Use consistently timezone-naive local timestamps, as in the verified extract. The loader reports incompatible timezone inputs as an error; it does not infer an institution’s timezone. Equal timestamps retain input order. An all-invalid/empty input fails with a descriptive error.

```bash
python -m cognishift --input /path/to/authorized/poe.csv --output-dir outputs/evaluation
```

At least five providers must have an eligible 15-order constructed shift. Each training fold must contain two classes after applying its own training-derived 75th-percentile cutoff; constant modification rates cannot support this evaluation. Input and output paths are relative to the caller’s working directory unless absolute.

The verified original extract contains 45,154 POE rows (44,362 retained). **Reproducing the original 0.865 ROC-AUC requires separately obtaining authorized MIMIC-IV data and the original extract.** Providing your own authorized MIMIC-IV copy enables the same evaluation protocol, but a different extract need not reproduce the published metrics. The current evaluation needs neither EMAR nor behavioral-fatigue data. The built-in synthetic demonstration runs without any of these CSVs and is not clinical performance evidence.

## Behavioral dataset license verification

`decision_fatigue_dataset.csv` is a local rename of `human_decision_fatigue_dataset.csv` from [Sonal Shinde's Human Decision Fatigue Behavioral Dataset](https://www.kaggle.com/datasets/sonalshinde123/human-decision-fatigue-behavioral-dataset), version 1. The source describes it as synthetic and explicitly lists **CC0: Public Domain**. [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) permits copying, modification, and redistribution, including commercial use.

Verified on **October 28, 2026**: the local CSV matches the file in the [official Kaggle version-1 download](https://www.kaggle.com/api/v1/datasets/download/sonalshinde123/human-decision-fatigue-behavioral-dataset?datasetVersionNumber=1) byte-for-byte: 1,522,913 bytes, 25,000 rows, and 13 columns. Its SHA-256 is:

```text
37fdc62d90dbecbe928e8cd4feed92bcb86cdeea41bd1b127db26ee6dd8b5d75
```

Only this verified behavioral CSV is excepted from the data-directory ignore rules. It is included for provenance and continuity with the original project; it contributes neither predictors nor labels to the reported evaluation. Its CC0 terms do not apply to MIMIC-IV or other project files.

## Provenance and publishing

The original POE and EMAR files include patient, admission, provider, and order identifiers. They are absent from the current repository tree and working data directory. Inputs under `data/` are ignored except this schema and the explicitly verified CC0 behavioral CSV. Do not add extracts, local predictions, or trained artifacts to a public repository without verifying their applicable sharing terms.

[PhysioNet’s MIMIC-IV documentation](https://physionet.org/content/mimiciv/3.1/) describes credentialed access and treats derived datasets/models as sensitive. It also explains that patient-specific date shifting makes dates incomparable across different patients. These constraints inform the project’s [chronology limitations](../docs/methodology.md#chronology-audit); the project does not claim to have verified de-identification of its local copies.

**This release continues the existing repository history.** Earlier commits retain historical input datasets and provider-level outputs; the current tree removes these files without rewriting those commits. Their retention does not grant readers permission to reuse or redistribute restricted data: obtain separate authorized access under the applicable terms. Private index/workspace snapshots and recoverability archives remain Git-ignored and must not be committed.

Published evidence in `docs/results/` is limited to aggregate fold metrics, an aggregate chart, and run metadata, with no patient/provider records or trained models. The [MIMIC-IV Data Use Agreement](https://physionet.org/content/mimiciv/view-dua/3.1/) contemplates open dissemination of research results and code; this does not grant permission to redistribute restricted datasets or row-level derivatives. The synthetic generator and CI never read clinical files.
