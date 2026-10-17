# Running the ENEL pipeline in Docker

The image contains Python 3.12 and every dependency of `pipeline.py`, so nothing has to be installed on the host
except Docker. The pipeline is the whole chain in one command:

real ENEL feeder → measured power wired in → parameters fitted → synthetic feeder generated →
statistical validation (KL divergence) → electrical validation (AC power flow) → synthetic feeder saved.

## Build

```bash
docker build -t gridstudio-enel .
```

## Run

The ENEL data is provided by ENEL and is **not** part of the image or the repository. Mount the folder that contains
`ppnet_DW00.1.382280-VE_202412/` and `timeseries_data/` at `/data` (read-only), and a folder for the result at `/output`.

PowerShell (Windows):

```powershell
docker run --rm `
  -v "C:\path\to\ENEL_grid_data:/data:ro" `
  -v "${PWD}\output:/output" `
  gridstudio-enel
```

Linux / macOS:

```bash
docker run --rm -v /path/to/ENEL_grid_data:/data:ro -v "$PWD/output:/output" gridstudio-enel
```

The generated feeder is written to `output/synthetic_enel_feeder.json` (pandapower JSON, load it with `pandapower.from_json`).

## Quick check (no data needed)

```bash
docker run --rm gridstudio-enel --help
```

If this prints the list of options, the image itself works. Running without mounted data stops immediately with a
clear message ("ENEL network file not found ..."), it does not crash.

## Troubleshooting

| Message / symptom | Cause and fix |
|---|---|
| `ENEL network file not found: /data/ppnet_.../...json` | The data folder is not mounted, or it is the wrong folder. `-v` must point to the folder that **contains** `ppnet_DW00.1.382280-VE_202412/` and `timeseries_data/`. |
| `error during connect` / `cannot connect to the Docker daemon` | Docker Desktop is not running yet. Start it and wait until it reports "Engine running". |
| Path with spaces on Windows | Put the whole `-v` argument in double quotes, as in the example above. |
| No `synthetic_enel_feeder.json` appears | The output folder was not mounted. Add `-v "${PWD}\output:/output"` (PowerShell) or `-v "$PWD/output:/output"` (Linux/macOS). |

## Options

Extra arguments after the image name are passed to `pipeline.py`:

```bash
docker run --rm -v ... gridstudio-enel --seed 7     # another random synthetic feeder
docker run --rm gridstudio-enel --help
```

| Option | Environment variable | Default in the container |
|---|---|---|
| `--enel-data-dir` | `ENEL_DATA_DIR` | `/data` |
| `--output-dir` | `OUTPUT_DIR` | `/output` |
| `--seed` | | `42` |

## Notes

- The dependency versions are pinned in `requirements-docker.txt` (the versions the pipeline was developed with).
- The source is copied into the image instead of being installed with `pip install .`, because `pyproject.toml` refers to
  `README.md` while the file is called `readme.md`. That works on Windows but fails on Linux.
