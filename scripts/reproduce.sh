#!/usr/bin/env bash
# Reproduce the results of the paper, stage by stage. Run from the repository root.
#
#   bash scripts/reproduce.sh <stage> [parallel jobs]
#
# Stages, in order (each one reads the outputs of the earlier ones):
#   pilot      train the pilot runs (control runs with seeds 1000+, used by the decidability analysis)
#   train      train the 240 main trajectories
#   replay     replay every placement, ablation, evaluation sample size and approver setting
#   rq2        composition failures (RQ2), observed heterogeneity and the cases Tiered missed
#   decisions  per-round decisions and non-passing checks of selected placements
#   decidability  design-time decidability analysis, predicted against observed
#   analysis   research questions RQ1, RQ3, RQ5 and the statistical tests
#   posthoc    post-hoc analyses
#   figures    figures and tables of the paper, into paper/
#   all        every stage above
# The overhead study (RQ4) is separate: see scripts/overhead.sh.
# Finished trajectories and replays are skipped when a stage is run again.
set -eo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv/bin/python}
JOBS=${2:-$(getconf _NPROCESSORS_ONLN)}
DATASETS=${DATASETS:-"acs isic"}
O=outputs
stage=${1:?usage: bash scripts/reproduce.sh <stage> [parallel jobs]}

simulate() {  # plan, output folder: one process per trajectory
  local n; n=$("$PY" -I -m raimon.cli grid --plan "$1" | wc -l)
  mkdir -p "$2"
  seq 0 $((n - 1)) | xargs -P "$JOBS" -I{} "$PY" -I -m raimon.cli simulate --plan "$1" --task {} --out "$2"
}

run() {
  case $1 in
    pilot) for ds in $DATASETS; do simulate configs/${ds}_pilot.json $O/pilot/$ds; done ;;
    train) for ds in $DATASETS; do simulate configs/$ds.json $O/trajectories/$ds; done ;;
    replay)
      for ds in $DATASETS; do
        mkdir -p $O/replay/$ds
        for d in $O/trajectories/$ds/*/; do
          n=$(basename "$d"); [[ -f $d/completion.json && ! -f $O/replay/$ds/$n.csv ]] && echo "$n"
        done | xargs -P "$JOBS" -I{} "$PY" -I -m raimon.cli replay --plan configs/$ds.json \
            --runs $O/trajectories/$ds --name {} --out $O/replay/$ds/{}.csv
        "$PY" -I -c "import glob, pandas as pd; f = sorted(glob.glob('$O/replay/$ds/*.csv')); \
pd.concat(map(pd.read_csv, f)).to_csv('$O/replay/$ds.csv', index=False); print('$ds', len(f), 'trajectories')"
      done ;;
    rq2) for ds in $DATASETS; do
        "$PY" -I -m raimon.soundness --plan configs/$ds.json --runs $O/trajectories/$ds \
            --replay $O/replay/$ds.csv --out $O/rq2/$ds --workers "$JOBS"; done ;;
    decisions) for ds in $DATASETS; do
        "$PY" -I -m raimon.decisions --plan configs/$ds.json --runs $O/trajectories/$ds \
            --out $O/decisions/$ds --workers "$JOBS"; done ;;
    decidability) for ds in $DATASETS; do
        "$PY" -I -m raimon.decidability --plan configs/$ds.json --pilot $O/pilot/$ds \
            --main $O/trajectories/$ds --out $O/decidability/$ds; done ;;
    analysis) "$PY" -I -m raimon.analysis --runs acs=$O/replay/acs.csv isic=$O/replay/isic.csv --out $O/analysis ;;
    posthoc) "$PY" -I -m raimon.posthoc --runs acs=$O/replay/acs.csv isic=$O/replay/isic.csv \
        --decisions acs=$O/decisions/acs isic=$O/decisions/isic \
        --trajectories acs=$O/trajectories/acs isic=$O/trajectories/isic \
        --decidability acs=$O/decidability/acs isic=$O/decidability/isic --out $O/posthoc ;;
    figures) "$PY" -I scripts/paper_assets.py --root . --out paper ;;
    *) echo "unknown stage $1"; exit 1 ;;
  esac
}

if [[ $stage == all ]]; then
  for s in pilot train replay rq2 decisions decidability analysis posthoc figures; do echo "== $s"; run $s; done
else
  run "$stage"
fi
