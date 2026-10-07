# ssbylw
# PurpleAir PM2.5 Quality Control and Humidity Correction

This repository contains the supplementary Python code used to process PurpleAir PA-II observations for a study of cold-weather evening PM2.5 enhancement in Leeds, UK.

The script converts station-level raw data into quality-controlled hourly observations, applies a relative-humidity correction and retains the three heating seasons included in the study.

## Scope

The script `supplementary_pm25_qc_humidity_correction.py` performs the following steps:

1. Reads all CSV files from a station-specific ZIP archive.
2. Checks that the required variables are present.
3. Applies physical-range checks to temperature and relative humidity.
4. Screens paired Plantower channels using a relative-difference criterion.
5. Aggregates valid observations to hourly means.
6. Applies the study’s humidity correction.
7. Retains observations from the 2022/23, 2023/24 and 2024/25 heating seasons.
8. Exports a corrected hourly CSV file.

The script processes one station per run. It should be run separately for the six study sites:

- SL001
- SL005
- SL006
- SL007
- SL017
- SL018

This code covers the quality-control and humidity-correction stages only. It does not reproduce the subsequent event classification, meteorological matching, Heating Degree Day calculation or binomial logistic-regression analysis described in the paper.

## Requirements

- Python 3.10 or later
- NumPy
- pandas

Install the required packages with:

```bash
python -m pip install numpy pandas
```

Using a virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Input data

Each run requires a ZIP archive containing the PurpleAir CSV exports for one station.

Each usable CSV file must contain the following columns exactly as written:

| Column | Description |
|---|---|
| `date` | Observation timestamp |
| `Temperature (F)` | Sensor temperature in degrees Fahrenheit |
| `Humidity (%)` | Sensor relative humidity |
| `PM2.5 A (CF=1) (ug/m3)` | Channel A PM2.5 measurement |
| `PM2.5 B (CF=1) (ug/m3)` | Channel B PM2.5 measurement |

CSV files smaller than 1,000 bytes are treated as empty and skipped by default.

The raw monitoring data are not included in this repository. Their use and redistribution may be subject to the conditions of the original data provider.

## Usage

Display the available options:

```bash
python supplementary_pm25_qc_humidity_correction.py --help
```

Example for station SL018:

```bash
python supplementary_pm25_qc_humidity_correction.py \
  --input-zip data/SL018.zip \
  --sensor-id SL018 \
  --output-dir outputs/corrected
```

Repeat the command for each station using the corresponding ZIP archive and station identifier.

The processing parameters can also be changed from the command line:

```bash
python supplementary_pm25_qc_humidity_correction.py \
  --input-zip data/SL018.zip \
  --sensor-id SL018 \
  --output-dir outputs/sensitivity \
  --kappa 0.41 \
  --rh-cap 85 \
  --ab-threshold 0.70
```

## Data processing

### Temperature and relative humidity

Temperature is converted from degrees Fahrenheit to degrees Celsius. Values outside −10 to 35 °C are treated as missing.

Relative-humidity values outside 0–100% are also treated as missing.

These checks do not automatically remove the complete PM2.5 observation. However, an hourly observation without a valid relative-humidity value cannot receive a corrected PM2.5 value.

### Paired-channel quality control

The mean PM2.5 concentration reported by channels A and B is calculated as:

```text
PM2.5_mean = (A + B) / 2
```

The relative channel difference is calculated as:

```text
relative_difference = |A - B| / (PM2.5_mean + 0.001)
```

An observation is retained when:

```text
relative_difference < 0.70
```

The small constant in the denominator prevents division by zero when both channels report values close to zero.

This analysis uses the 70% relative-difference component of the paired-channel quality-control approach described by Barkjohn et al. (2022). It does not reproduce every condition in the original Barkjohn quality-control rule.

### Hourly aggregation

Valid observations are grouped by UTC hour. The script calculates:

- mean relative humidity;
- mean temperature;
- mean uncorrected PM2.5;
- number of valid PM2.5 observations contributing to the hour.

### Humidity correction

Hours with a mean relative humidity of exactly 100% are excluded. Relative humidity is then capped at 85% by default.

Corrected PM2.5 is calculated as:

```text
PM2.5_corrected =
PM2.5_raw / [1 + kappa × RH_capped / (100 − RH_capped)]
```

The default parameters are:

```text
kappa = 0.38
RH cap = 85%
```

The value of κ was transferred from published optical-particle-counter research. It was not independently calibrated for every PurpleAir site or pollution episode. This limitation should be considered when interpreting the corrected concentrations.

### Heating seasons

The script retains observations from the following periods:

| Heating season | Included period |
|---|---|
| 2022/23 | 1 October 2022 to 31 March 2023 |
| 2023/24 | 1 October 2023 to 31 March 2024 |
| 2024/25 | 1 October 2024 to 31 March 2025 |

The intervals use 1 April as the exclusive end date so that the whole of 31 March is retained.

## Output

For a station identified as `SL018`, the output file is:

```text
outputs/corrected/SL018_corrected_hourly.csv
```

The output contains the following columns:

| Column | Description |
|---|---|
| `hour_utc` | Beginning of the hourly interval in UTC |
| `RH` | Hourly mean relative humidity (%) |
| `RH_capped` | Relative humidity after applying the upper cap (%) |
| `Temperature_C` | Hourly mean valid temperature (°C) |
| `pm25_raw` | Hourly mean paired-channel PM2.5 |
| `pm25_corrected` | Humidity-corrected hourly PM2.5 |
| `correction_factor` | Corrected PM2.5 divided by raw PM2.5 |
| `record_count` | Number of valid observations contributing to the hour |

PM2.5 concentrations are expressed in micrograms per cubic metre.

The script also prints:

- files read and skipped;
- observations retained after channel screening;
- number of hourly observations;
- mean PM2.5 before and after correction;
- heating-season summary statistics;
- location of the output file.

## Suggested folder structure

```text
project/
├── README.md
├── supplementary_pm25_qc_humidity_correction.py
├── data/
│   ├── SL001.zip
│   ├── SL005.zip
│   ├── SL006.zip
│   ├── SL007.zip
│   ├── SL017.zip
│   └── SL018.zip
└── outputs/
    └── corrected/
```

Raw data and derived outputs should not be uploaded to a public repository if their licence, consent conditions or file size prevent redistribution.

## Reproducibility notes

- Keep an unchanged copy of each input ZIP archive.
- Record the Python, NumPy and pandas versions used for the final analysis.
- Retain the processing log for each station.
- Use the same parameters for all six main-network stations unless conducting a documented sensitivity analysis.
- All timestamps are handled as UTC.
- Comparisons with local clock time must account for British Summer Time.
- Missing observations are not imputed.
- The script processes one station at a time.
- The script does not perform event detection or statistical modelling.

## References

Barkjohn, K.K. et al. (2022). Correction and accuracy of PurpleAir PM2.5 measurements for extreme wildfire smoke. *Sensors*. 22(24), 9669.  
https://doi.org/10.3390/s22249669

Crilley, L.R. et al. (2018). Evaluation of a low-cost optical particle counter (Alphasense OPC-N2) for ambient air monitoring. *Atmospheric Measurement Techniques*. 11, pp.709–720.  
https://doi.org/10.5194/amt-11-709-2018

Consult the paper and its appendices for the complete methodological justification, sensitivity analyses and reference list.

## Citation

Add the final article citation and repository DOI here after the paper or archived code release receives a permanent identifier.

## Licence

No software licence has yet been specified. A licence file should be added before the code is distributed publicly.
