#!/usr/bin/env bash
# Sequential training queue for the composition experiments.
# Runs each job back-to-back: a job starts only after the previous one finishes.
# Pure bash + yolo CLI -> once launched (tmux or nohup) it needs NObody connected.
#
# Launch it detached, e.g.:
#   tmux new -s trainq        # then run: bash scripts/train_queue.sh
#   (detach: Ctrl-b then d ; reattach: tmux attach -t trainq)
# or:
#   nohup bash scripts/train_queue.sh > runs/train/queue.log 2>&1 &
#
# Watch overall progress:  tail -f runs/train/queue.log
# Watch a single job:      tail -f runs/train/<NAME>_train.log
set -u

ROOT=/home/dell/cetacean-detection-final
YOLO="$ROOT/ml-venv/bin/yolo"
PROJECT="$ROOT/runs/train"          # absolute -> keeps runs in THIS project
EPOCHS=100
IMGSZ=1024
SEED=1337
PATIENCE=30
N_BATCH=18                          # batch for yolo11n jobs
S_BATCH=10                           # batch for the yolo11s job (adjust after VRAM test)

mkdir -p "$PROJECT"

# job format: NAME | MODEL | DATA_YAML | BATCH
JOBS=(
  "E0_s|yolo11s.pt|$ROOT/data/dataset/B/data.yaml|$S_BATCH"
  "E1|yolo11n.pt|$ROOT/data/dataset/E1/data.yaml|$N_BATCH"
  "E2|yolo11n.pt|$ROOT/data/dataset/E2/data.yaml|$N_BATCH"
  "E4_0|yolo11n.pt|$ROOT/data/dataset/E4_0/data.yaml|$N_BATCH"
  "E4_20|yolo11n.pt|$ROOT/data/dataset/E4_20/data.yaml|$N_BATCH"
)

log() { echo "[$(date '+%F %T')] $*"; }

for job in "${JOBS[@]}"; do
  IFS='|' read -r NAME MODEL DATA BATCH <<< "$job"
  results="$PROJECT/$NAME/results.csv"

  # skip if this job already finished all EPOCHS
  if [[ -f "$results" ]] && [[ "$(tail -n +2 "$results" | wc -l)" -ge "$EPOCHS" ]]; then
    log "SKIP $NAME (already has >= $EPOCHS epochs)"
    continue
  fi

  log "START $NAME  model=$MODEL batch=$BATCH data=$DATA"
  "$YOLO" detect train \
    model="$MODEL" \
    data="$DATA" \
    imgsz="$IMGSZ" epochs="$EPOCHS" batch="$BATCH" device=0 \
    seed="$SEED" patience="$PATIENCE" \
    project="$PROJECT" name="$NAME" exist_ok=True \
    > "$PROJECT/${NAME}_train.log" 2>&1
  rc=$?

  if [[ $rc -ne 0 ]]; then
    log "FAIL $NAME (exit $rc) -> see $PROJECT/${NAME}_train.log ; STOPPING queue"
    exit $rc
  fi
  log "DONE  $NAME -> $PROJECT/$NAME/weights/best.pt"
done

log "QUEUE COMPLETE (E0 yolo11m still pending - run at end of phase)"
