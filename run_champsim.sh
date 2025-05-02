#!/usr/bin/env bash
set -euo pipefail

TRACE_DIR="traces"
WARMUP=5000
SIM=50000
BIN="./bin/champsim"
SUMMARY="nn_summary.csv"

echo "trace,total,mispredict,accuracy" > "${SUMMARY}"
echo "Evaluating NN predictor on all traces in ${TRACE_DIR}"

for TRACE in "${TRACE_DIR}"/*.champsimtrace.xz; do
  NAME=$(basename "${TRACE}" .champsimtrace.xz)
  echo "---- ${NAME} ----"

  OUTPUT="$("${BIN}" \
    --warmup-instructions "${WARMUP}" \
    --simulation-instructions "${SIM}" \
    "${TRACE}" 2>&1)"

  TOTAL=$(printf '%s\n' "${OUTPUT}" \
    | grep -m1 "Simulation finished" \
    | awk '{print $6}')

  ACCPCT=$(printf '%s\n' "${OUTPUT}" \
    | grep -m1 "Branch Prediction Accuracy:" \
    | awk '{print $6}' \
    | tr -d '%')

  if [[ -z "${TOTAL}" || -z "${ACCPCT}" ]]; then
    echo "Warning: failed to parse stats for ${NAME}"
    continue
  fi

  MIS=$(
    awk -v t="${TOTAL}" -v a="${ACCPCT}" \
      'BEGIN{ printf "%.0f", t*(1-(a/100)) }'
  )

  echo "${NAME},${TOTAL},${MIS},${ACCPCT}" >> "${SUMMARY}"
done

echo "Done: see per-trace summary in ${SUMMARY}"
