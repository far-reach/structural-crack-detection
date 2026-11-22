#!/usr/bin/env bash
# Full vision experiment queue (sequential, one GPU). Run from the repository root.
# Set PYTHON to the interpreter of your environment if it is not on the path.
set -u
PY=${PYTHON:-python}
export PYTHONIOENCODING=utf-8
run() { echo "=== $*"; $PY "$@" 2>&1 | grep -v -i -E "warn|symlink|sched.step" | grep -E "TEST|params|Traceback|Error|acc|seed|tile|grouped|v1 on|\{" ; }
for s in 0 1 2; do run experiments/train_codebrim.py --model convnext_tiny --epochs 30 --lr 2e-4 --seed $s; done
for s in 0 1 2; do
  run experiments/train_codebrim.py --model resnet50 --epochs 30 --lr 3e-4 --seed $s
  run experiments/train_codebrim.py --model mobilenetv3_small_100 --epochs 40 --lr 1e-3 --seed $s
  run experiments/train_codebrim.py --model mobilenetv3_small_100 --epochs 40 --lr 1e-3 --seed $s --teacher convnext_tiny_r224_s$s
  run experiments/train_codebrim.py --model nano32 --epochs 60 --lr 3e-3 --seed $s
  run experiments/train_codebrim.py --model nano32 --epochs 60 --lr 3e-3 --seed $s --teacher convnext_tiny_r224_s$s
  run experiments/train_codebrim.py --model nano16 --epochs 60 --lr 3e-3 --seed $s
  run experiments/train_codebrim.py --model nano16 --epochs 60 --lr 3e-3 --seed $s --teacher convnext_tiny_r224_s$s
done
echo QUEUE_A_DONE
run experiments/leak_study.py --seeds 0 1 2
echo QUEUE_DONE
