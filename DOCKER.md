<!-- Author: Adam Lahmar <adam.lahmar@rwth-aachen.de> -->
# Running the pipeline with Docker

The Docker image contains Python 3.12 and every package the pipeline needs, so **nothing has to be installed on your computer except Docker**.
One command does the whole chain:

real ENEL feeder → measured power wired in → parameters fitted → synthetic feeder generated → statistical validation → electrical validation (AC power flow)
→ synthetic feeder saved → **interactive map of the real and the synthetic feeder written**.

## 1. What you need

* **Docker Desktop** (Windows, macOS) or Docker Engine (Linux), running. Check with `docker --version`.
* **This repository** (clone or download it).
* **The ENEL data**: the folder that contains `ppnet_DW00.1.382280-VE_202412/` and `timeseries_data/`. It is provided by ENEL and is **not** part of the
  repository or of the image. It is mounted read-only at `/data`; nothing is written to it.

## 2. Build the image (once)

In a terminal **in the repository folder** (the one that contains the `Dockerfile`):

```bash
docker build -t gridstudio-enel .
```

## 3. Run it

Mount the data folder at `/data` (read-only) and a folder for the results at `/output`.

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

The run takes about a minute and prints the eight steps of the pipeline (structure of the real feeder, fitted parameters, comparison of the synthetic with the real
feeder, power flow, and the paths of the saved files).

## 4. See the results

Everything is written to the `output` folder (the folder you mounted at `/output`):

| File | What it is |
|---|---|
| **`synthetic_enel_map.html`** | **Open this in a web browser** (double-click it, or `start output\synthetic_enel_map.html` in PowerShell). One self-contained page, no server and no internet needed. |
| `synthetic_enel_feeder.json` | The synthetic feeder as a pandapower network (`pandapower.from_json`): grid connection, transformer, breaker, cables and switches, loads and generators. |

**What the map shows.** Both networks side by side (button *Real network* / *Synthetic network*), with the equipment drawn explicitly: the grid connection, the transformer, the
20 kV busbar, the breaker, the cables (thickness = current rating) and the switches (dashed). Three operating points (annual mean, the highest load and the highest generation of
your data) and three colourings (bus type, voltage, cable loading) show how the feeder behaves. Click a bus to trace its route to the grid connection and to see its voltage;
click a cable, a switch or the transformer for its data and loading. Scroll to zoom, drag to move.

**The map contains data of your real feeder. Treat it like the data itself and do not publish it.**

## 5. Options

Extra arguments after the image name are passed on to the pipeline:

```bash
docker run --rm -v ... gridstudio-enel --seed 7        # another random synthetic feeder
docker run --rm -v ... gridstudio-enel --no-map        # without the map
docker run --rm gridstudio-enel --help                 # all options (no data needed)
```

| Option | Effect |
|---|---|
| `--seed N` | random seed of the synthetic feeder (default 42; the same seed always gives the same feeder) |
| `--no-map` | do not write the map page |
| `--enel-data-dir`, `--output-dir` | folders inside the container (defaults `/data` and `/output`; normally not needed) |
| `--feeder-only --default-fit --mean-sizing --default-cables` | together: the first version that was delivered (every connection a cable, no transformer, no switches, defaults of the paper) |

Each of the four last options can also be used alone to switch one improvement off: `--feeder-only` (no transformer/switches), `--default-fit` (defaults of the paper for
shares, positions and cable-length cut-off), `--mean-sizing` (cables sized for the annual average only), `--default-cables` (default Dutch cable library instead of the cable types
of the real network).

## 6. Optional: comparison with the real feeder and the tests

```bash
# real feeder against 10 synthetic feeders: structure, cables and one real day (10-20 minutes); table on screen, numbers in output/report_rows.json
docker run --rm -v "<ENEL data folder>:/data:ro" -v "<output folder>:/output" gridstudio-enel compare --out /output/report_rows.json

# automated tests (no data needed)
docker run --rm gridstudio-enel test

# list of the commands
docker run --rm gridstudio-enel help
```

## 7. Shortcut with docker compose (optional)

Instead of typing the folders every time: copy `.env.example` to `.env`, set `ENEL_DATA_DIR` (forward slashes, also on Windows), then

```bash
docker compose --profile tools build
docker compose run --rm pipeline        # = section 3; results in the folder output/
docker compose run --rm compare         # = comparison of section 6
docker compose run --rm test            # = tests of section 6
```

## 8. Troubleshooting

| Message / symptom | Cause and fix |
|---|---|
| `ENEL network file not found: /data/ppnet_.../...json` | The data folder is not mounted, or it is the wrong folder. `-v` must point to the folder that **contains** `ppnet_DW00.1.382280-VE_202412/` and `timeseries_data/`. |
| `error during connect` / `cannot connect to the Docker daemon` | Docker Desktop is not running yet. Start it and wait until it reports "Engine running". |
| Path with spaces on Windows | Put the whole `-v` argument in double quotes, as in the example. |
| No files in `output` | The output folder was not mounted. Add `-v "${PWD}\output:/output"` (PowerShell) or `-v "$PWD/output:/output"` (Linux/macOS). |
| No `synthetic_enel_map.html` | You used `--no-map`, or look in the folder you mounted at `/output`. The pipeline prints the path of the file. |
| `no configuration file provided` (compose) | Run the command in the repository folder (the one with `docker-compose.yml`). |

## 9. Notes

* The dependency versions are pinned in `requirements-docker.txt` (the versions the work was developed with). The image uses Python 3.12.
* The source is copied into the image instead of being installed with `pip install .`, because `pyproject.toml` refers to `README.md` while the file is called
  `readme.md`. That works on Windows but fails on Linux.
* Results are reproducible: the same seed gives the same feeder on every computer (the content is identical; the bytes of the JSON file can differ slightly between Python versions).
* `test` runs about 240 tests of the distribution pipeline and the map. Four test modules that need the optional package `pypowsybl` (export of the transmission-grid part)
  are skipped in the image: the pipeline does not use it.
