#!/bin/bash
set -e

#configuration
TRACE_DIR="traces"
WARMUP=100000
SIM=900000
BIN="./bin/champsim"
OUTPUT_DIR="csv_output"
FINAL_OUTPUT="combined_gather.csv"

# Prepare output
mkdir -p "$OUTPUT_DIR"
> "$FINAL_OUTPUT"

echo "Running ChampSim on all traces in $TRACE_DIR..."

#loop over every .xz trace
for TRACE in "$TRACE_DIR"/*.champsimtrace.xz; do
    BASENAME=$(basename "$TRACE" .champsimtrace.xz)
    echo "----------------------------"
    echo "Running trace: $BASENAME"
    echo "----------------------------"

    #run ChampSim with custom instruction limits
    $BIN --warmup_instructions "$WARMUP" \
         --simulation_instructions "$SIM" \
         "$TRACE"

    #move output file
    OUT_CSV="gather_more.csv"
    if [ -f "$OUT_CSV" ]; then
        mv "$OUT_CSV" "$OUTPUT_DIR/$BASENAME.csv"
    else
        echo "Warning: $OUT_CSV not found for $TRACE"
    fi
done

echo "Merging CSV outputs into $FINAL_OUTPUT..."

#concatenate all CSVs, preserving only the first header
FIRST=1
for CSV in "$OUTPUT_DIR"/*.csv; do
    if [ $FIRST -eq 1 ]; then
        cat "$CSV" >> "$FINAL_OUTPUT"
        FIRST=0
    else
        tail -n +2 "$CSV" >> "$FINAL_OUTPUT"
    fi
done

echo "Done! Final output written to: $FINAL_OUTPUT"
