#!/usr/bin/env bash
# Runs the same Locust flood against 1, 2, 3 and 4 API containers and stores CSVs in results/.
# Usage: scripts/run_load_tests.sh [USERS=50] [SPAWN_RATE=10] [DURATION=60s]
set -euo pipefail
USERS=${1:-50}; RATE=${2:-10}; DUR=${3:-60s}
mkdir -p results
for N in 1 2 3 4; do
  echo "=== $N container(s) ==="
  docker compose down
  docker compose up -d --build --scale api=$N --wait
  sleep 15                                        # let every replica load the model
  locust -f locustfile.py --host http://localhost:8080 --headless \
         -u "$USERS" -r "$RATE" -t "$DUR" --csv "results/containers_$N" --only-summary
done
docker compose down
python scripts/summarize_load.py
