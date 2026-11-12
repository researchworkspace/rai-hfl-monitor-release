# Statistical Monitoring of Responsible-AI Requirements in Hierarchical Federated Learning

Code and instructions to reproduce the results of the paper. The method, called *Tiered* in the paper, checks
each Responsible-AI requirement of a client → edge → cloud federation at the tier that holds its evidence,
turns confidence intervals into three-valued verdicts (pass, fail, undecided), and decides deployment per domain
with an Open Policy Agent (Rego) policy.

## Layout

| Path | Content |
|---|---|
| `src/raimon/` | simulator, learners, injections, interval statistics, replay engine, analyses |
| `policies/` | the Rego decision policy (`verdict.rego`) and its unit tests (`verdict_test.rego`) |
| `configs/` | run plans: `acs.json` and `isic.json` (main experiment), `acs_pilot.json` and `isic_pilot.json` (pilot runs) |
| `scripts/reproduce.sh` | every stage of the main experiment, from training to figures and tables |
| `scripts/overhead.sh`, `overhead/Dockerfile` | overhead study (RQ4) with emulated edge and cloud nodes |
| `scripts/get_data.sh` | download of the dermatology data and computation of its embeddings |
| `scripts/paper_assets.py` | figures and tables of the paper, from `outputs/` |
| `tests/` | unit tests |

Names in the code and in the paper:

| Code | Paper |
|---|---|
| `acs`, `isic` | census federation, dermatology federation |
| P1, P2, P2s, P2c, P3, P4, P4R | Central, Uniform, Uniform-scoped, Uniform+clients, Pooled, Tiered, Tiered-restricted |
| R1 … R7 | statistical-parity gap, equal-opportunity gap, worst-group accuracy, privacy budget, robustness, explanation stability, report integrity |
| V0 … V6 | control, domain drift, Simpson shift, poisoning, lying clients, noise cut, acquisition shift |
| A1, A2a, A2b, A3, A4 | point estimates, undecided deploy, undecided hold, unadjusted local threshold, no integrity alarm |
| `fedavg`, `fairfed`, `eafclc` | FedAvg, FairFed, EA-FCL-c |
| `reviewer` (setting mode) | approver |
| `e0`, `e1`, … | domains (edges): census divisions, dermatology centres |

## Environment

The results in the paper were obtained with Python 3.11 and the package versions pinned in `pyproject.toml`
(PyTorch 2.15.0, Opacus 1.6.0, NumPy 2.4.6, among others) and OPA 1.22.2.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Install [OPA 1.22.2](https://www.openpolicyagent.org/docs/latest/#running-opa) so that `opa` is on `PATH`; the
replay starts a local `opa run --server` loaded with `policies/`. Docker is needed for the overhead study only.

Check the installation:

```bash
opa test policies/
.venv/bin/python -m pytest -q
```

The scripts use `.venv/bin/python`; set `PYTHON` to use another interpreter.

## Data

- **Census (ACS 2018 one-year, income task).** `folktables` downloads it into `data/acs/` on first use.
- **Dermatology (Fed-ISIC2019).** `bash scripts/get_data.sh` downloads FLamby's centre split and the ISIC 2019
  images and metadata into `data/isic/raw/` (about 9.8 GB) and computes the embeddings of a frozen ImageNet
  EfficientNet-B0 once (`data/isic/embeddings.npz`; about 4 minutes on an Apple M4 Max). Set `DEVICE=cuda` or
  `DEVICE=cpu` to choose the device.

The plans read the data from `data/`; set `RAIHFL_DATA_ROOT` to use another location.

## Reproducing the results

Every stage runs from the repository root, writes into `outputs/` and reads the outputs of the earlier stages.
The second argument is the number of parallel processes (default: all cores).

```bash
bash scripts/reproduce.sh all            # every stage below, in order
bash scripts/reproduce.sh <stage> [jobs] # one stage
```

| Stage | What it does | Output |
|---|---|---|
| `pilot` | 9 pilot runs (control runs with seeds 1000 and 1001), used by the decidability analysis only | `outputs/pilot/` |
| `train` | 240 trajectories: 2 federations × 3 learners × 8 injection settings × 5 seeds | `outputs/trajectories/` |
| `replay` | every placement, ablation, evaluation sample size and approver setting (38 per trajectory) | `outputs/replay/{acs,isic}.csv` |
| `rq2` | composition failures (RQ2), observed heterogeneity, the cases Tiered missed | `outputs/rq2/` |
| `decisions` | per-round decisions and non-passing checks of selected placements | `outputs/decisions/` |
| `decidability` | design-time decidability analysis, predicted against observed | `outputs/decidability/` |
| `analysis` | RQ1, RQ3 and RQ5 and the statistical tests (exact and cluster sign-flip, Holm) | `outputs/analysis/` |
| `posthoc` | post-hoc analyses: decision rules, localisation, imperfect approver, coverage, model in service, sample size | `outputs/posthoc/` |
| `figures` | figures and tables of the paper | `paper/figures/`, `paper/tables/` |

Training is the expensive stage. In the paper, a census trajectory took a median of 543 s on 16 CPU cores with
96 GB of memory (one trajectory per job on an HPC cluster), and a dermatology trajectory a median of 59 s on an
Apple M4 Max with 64 GB. To train on a cluster, run one trajectory per array task:

```bash
.venv/bin/python -I scripts/raimon_cli.py grid --plan configs/acs.json        # lists the 120 trajectories
.venv/bin/python -I scripts/raimon_cli.py simulate --plan configs/acs.json --task $SLURM_ARRAY_TASK_ID \
    --out outputs/trajectories/acs
```

Finished trajectories and replays are skipped when a stage is run again. `DATASETS=isic bash
scripts/reproduce.sh <stage>` restricts a stage to one federation.

Trajectories are deterministic given the seed: on the same hardware and package versions, rerunning a
trajectory reproduces every logged array except the wall-clock times, and the replay and everything after it is
deterministic given the trajectories. Results on other hardware can differ in floating-point detail.

### Overhead (RQ4)

Run on an otherwise idle machine with Docker, after the `pilot` and `train` stages:

```bash
bash scripts/overhead.sh workloads   # client counts and statistic sizes from logged trajectories
bash scripts/overhead.sh mlp         # census MLP and dermatology head, 3 repetitions
bash scripts/overhead.sh full        # dermatology with the full EfficientNet-B0
bash scripts/overhead.sh report      # outputs/overhead/{mlp,full}/summary.json
```

Each edge is a container limited to one CPU, 2 GB of memory and a 20 Mbit/s uplink with 25 ms delay; the cloud has
four CPUs and 4 GB. Absolute times depend on the host; the paper reports them for an Apple M4 Max (Docker with a
Linux virtual machine of 16 CPUs and 8 GB).

### Figures and tables

`bash scripts/reproduce.sh figures` writes every data figure and table of the paper (Figures 1 bottom and 3–9,
all tables) into `paper/`; `.venv/bin/python -I scripts/paper_assets.py --root . --out paper --only <job> …`
regenerates selected ones. The schematic diagrams (the deployment cases in Figure 1 and the method overview in
Figure 2) are drawn by hand and are not generated.

| Paper item | File |
|---|---|
| Figure 1 (bottom), motivating cases on real data | `fig_motivation.pdf` |
| answers to RQ1 and RQ3 | `fig_exposure.pdf`, `table_exposure.tex`, `table_rq1.tex`, `table_rq3.tex`, `table_delay.tex` |
| answer to RQ2 | `fig_soundness.pdf`, `table_rq2.tex`, `table_beta.tex` |
| answers to RQ4 and RQ5 | `fig_overhead.pdf`, `table_overhead.tex`, `fig_decidability.pdf` |
| Tiered without an approver and selected decision rules | `fig_tradeoff.pdf`, `table_ph1.tex`, `fig_timeline.pdf` |
| which mechanisms matter | `table_ablation.tex` |
| localisation | `table_localisation.tex`, `table_integrity.tex` |
| an imperfect approver | `fig_reviewer.pdf`, `table_reviewer.tex` |
| predicting decidability | `table_decidability.tex`, `table_samplesize.tex` |
| coverage of the intervals | `table_coverage.tex` |
| the model in service | `table_retained.tex` |
| where Tiered fails | `table_misses.tex` |

`paper/assets_facts.json` collects further numbers quoted in the text.

## Calibration

The thresholds in `configs/*.json` were placed just above the range that the control runs of the pilot reached.
`python -I -m raimon.calibrate --runs outputs/pilot/acs` prints the levels of every monitored statistic in the
pilot runs.

## License

Licensed under either of the [MIT License](LICENSE-MIT) or the [Apache License, Version 2.0](LICENSE-APACHE), at
your option. The datasets are not part of this repository and keep their own terms: the ACS data of the US Census
Bureau and the ISIC 2019 challenge data used through FLamby.
