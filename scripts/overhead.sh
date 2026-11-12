#!/usr/bin/env bash
# Overhead study (RQ4): emulated edge nodes (1 CPU, 2 GB, 20 Mbit/s uplink with 25 ms delay) and a cloud node
# (4 CPUs, 4 GB) as Docker containers on one host. Run from the repository root, with nothing else running.
#
#   bash scripts/overhead.sh workloads   build the workloads from logged trajectories (needs the pilot and train stages)
#   bash scripts/overhead.sh mlp         measure the census MLP and the dermatology head, 3 repetitions
#   bash scripts/overhead.sh full        measure the dermatology federation with the full EfficientNet-B0
#   bash scripts/overhead.sh report      summarise both measurements
#
# Outputs: outputs/overhead/{mlp,full}/<dataset>/rep<r>/{cloud.json,edge<e>.json,train.json} and summary.json.
set -eo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv/bin/python}
O=outputs/overhead
REPS=${OVERHEAD_REPS:-3}
NETEM="delay 25ms rate 20mbit"
IMG=raimon-node:latest

measure() {  # output folder, datasets
  local out=$1; shift
  docker info >/dev/null 2>&1 || { echo "ERROR: Docker daemon not running"; exit 1; }
  docker build -q -t "$IMG" -f overhead/Dockerfile . >/dev/null
  docker network inspect raimon-ovh >/dev/null 2>&1 || docker network create raimon-ovh >/dev/null
  mkdir -p "$out"
  docker run --rm --entrypoint python "$IMG" -I -c "
import json, platform, subprocess, numpy, opacus, torch
print(json.dumps({'netem': '$NETEM', 'reps': $REPS, 'platform': platform.platform(), 'torch': torch.__version__,
  'opacus': opacus.__version__, 'numpy': numpy.__version__,
  'opa': subprocess.run(['opa', 'version'], capture_output=True, text=True).stdout.splitlines()[0]}, indent=1))" \
    > "$out/environment.json"
  cleanup() { docker rm -f $(docker ps -aq --filter name=raimon-ovh-) >/dev/null 2>&1 || true; }
  trap cleanup EXIT
  for ds in "$@"; do
    local wl=$O/workloads/$ds.npz
    [[ -f $wl ]] || { echo "ERROR: missing $wl (run the workloads step)"; exit 1; }
    E=$("$PY" -I -c "import numpy as np; print(int(np.load('$wl')['client_edge'].max()) + 1)")
    for r in $(seq 1 "$REPS"); do
      d=$PWD/$out/$ds/rep$r
      if [[ -f $d/completion.json ]]; then echo "skip $ds rep$r (complete)"; continue; fi
      mkdir -p "$d"
      cleanup
      docker run -d --name raimon-ovh-cloud --network raimon-ovh --cpus=4 --memory=4g -e RAIMON_THREADS=4 \
        -v "$PWD/$O/workloads":/w:ro -v "$d":/out "$IMG" overhead cloud --workload /w/$ds.npz --port 8080 \
        --out /out/cloud.json >/dev/null
      until docker logs raimon-ovh-cloud 2>&1 | grep -q listening; do sleep 0.5; done
      for e in $(seq 0 $((E - 1))); do
        docker run -d --name raimon-ovh-edge$e --network raimon-ovh --cpus=1 --memory=2g --cap-add NET_ADMIN \
          -v "$PWD/$O/workloads":/w:ro -v "$d":/out --entrypoint sh "$IMG" -c \
          "tc qdisc add dev eth0 root netem $NETEM && exec python -I /app/scripts/raimon_cli.py overhead edge --workload /w/$ds.npz --edge $e --cloud http://raimon-ovh-cloud:8080 --out /out/edge$e.json" >/dev/null
      done
      while true; do   # abort as soon as a node fails, otherwise wait until every node has exited
        bad=$(docker ps -a --filter name=raimon-ovh- --filter status=exited --format '{{.Names}} {{.Status}}' | grep -v 'Exited (0)' || true)
        [[ -z $bad ]] || { echo "ERROR: $bad"; exit 1; }
        [[ -z $(docker ps -q --filter name=raimon-ovh-) ]] && break
        sleep 2
      done
      cleanup
      docker run --rm --name raimon-ovh-train --cpus=1 --memory=2g -v "$PWD/$O/workloads":/w:ro -v "$d":/out "$IMG" \
        overhead train --workload /w/$ds.npz --out /out/train.json >/dev/null
      echo '{"complete": true}' > "$d/completion.json"
      echo "done $ds rep$r"
    done
  done
}

case ${1:?usage: bash scripts/overhead.sh workloads|mlp|full|report} in
  workloads)
    mkdir -p $O/workloads
    "$PY" -I -m raimon.cli overhead make-workload --plan configs/acs.json \
      --trajectory outputs/pilot/acs/acs__fedavg__V0__s1000 --out $O/workloads/acs.npz
    "$PY" -I -m raimon.cli overhead make-workload --plan configs/isic.json \
      --trajectory outputs/trajectories/isic/isic__fedavg__V0__s1 --out $O/workloads/isic.npz
    "$PY" -I -m raimon.cli overhead make-workload --plan configs/isic.json --arch effnet --rounds 2 \
      --trajectory outputs/trajectories/isic/isic__fedavg__V0__s1 --out $O/workloads/isic-full.npz ;;
  mlp) measure $O/mlp acs isic ;;
  full) measure $O/full isic-full ;;
  report)
    "$PY" -I -m raimon.overhead_report --root $O/mlp --out $O/mlp/summary.json
    "$PY" -I -m raimon.overhead_report --root $O/full --out $O/full/summary.json ;;
  *) echo "unknown step $1"; exit 1 ;;
esac
