#!/bin/bash

export LC_CTYPE=C.UTF-8

if [ "$1" = "train-cifar10" ]; then
  docker-compose -f ./docker/docker-compose-gpu.yml run --rm -T experiment python3 scripts/main.py local classifier --d.source CIFAR10 -n DIM_CIFAR10 --t.epochs 1000 --t.no_ascii

elif [ "$1" = "val-cifar10" ]; then
  docker-compose -f ./docker/docker-compose-gpu.yml run --rm -T experiment python3 scripts/main.py classifier --eval --d.source CIFAR10 -n DIM_CIFAR10_cls --t.eval_only -L .cortex/output/DIM_CIFAR10/binaries/DIM_CIFAR10_final.t7 --t.no_ascii

elif [ "$1" = "dim-neg-local" ]; then
  docker-compose -f ./docker/docker-compose-gpu.yml run --rm -T experiment python3 scripts/main.py glob local classifier --d.source CIFAR10 -n DIM_CIFAR10_neglocal --t.epochs 1000 --t.no_ascii --local.mode fd_neg

fi
