#!/usr/bin/env bash
# Restart the demo service on the VM, writing the handle aliases before it starts (the running
# service only reads them at startup).
#
#   scripts/start_demo.sh            keep the database
#   scripts/start_demo.sh --fresh    wipe the database first, for clean totals in a recording
set -euo pipefail

FRESH=0
[[ "${1:-}" == "--fresh" ]] && FRESH=1
DATA=~/shared/data
ENV_FILE=~/forest.env
RUN=(sudo docker run --network host --user "$(id -u):$(id -g)" -e HOME=/tmp
     --env-file "$ENV_FILE" -v ~/Forest:/app -v "$DATA":/app/data -v ~/shared/models:/models
     -w /app)

if [[ ! -f "$ENV_FILE" ]]; then
  # every stored hash and encrypted message depends on these keys, so never make new ones here
  echo "$ENV_FILE is missing: restore it, don't regenerate it" >&2
  exit 1
fi

sudo docker rm -f forest-api >/dev/null 2>&1 || true
if (( FRESH )); then
  rm -f "$DATA/forest.db" "$DATA/forest.db-wal" "$DATA/forest.db-shm"
  echo "database wiped"
fi
"${RUN[@]}" --rm forest:dev python -m scripts.make_aliases
"${RUN[@]}" -d --name forest-api --restart unless-stopped forest:dev >/dev/null

echo -n "waiting for forest-api"
for _ in $(seq 120); do
  if health=$(curl -sf http://127.0.0.1:8000/health); then
    echo; echo "$health"
    exit 0
  fi
  echo -n "."
  sleep 1
done
echo; echo "forest-api did not come up, see: sudo docker logs forest-api" >&2
exit 1
