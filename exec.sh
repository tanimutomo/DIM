#!/bin/bash

export LC_CTYPE=C.UTF-8

# main command
docker="docker-compose -f ./docker/docker-compose-gpu.yml run --rm -T experiment"
main_cmd="python3 scripts/main.py"

if [ "$1" = "docker" ]; then
  cmd="${docker} ${main_cmd}"
else
  cmd="${main_cmd}"
fi


# Basic options
name="DIM_CIFAR10_$2"
base_options="--d.source CIFAR10 --t.no_ascii --device $3 -n ${name}"


# Other options
if [ "$2" = "train-cifar10" ]; then
  mode="local classifier"
  options="--t.epochs 1000"

elif [ "$2" = "val-cifar10" ]; then
  mode="classifier"
  options="--eval --t.eval_only -L ~/.cortex/output/DIM_CIFAR10/binaries/DIM_CIFAR10_final.t7"

elif [ "$2" = "neg-local" ]; then
  mode="glob local classifier"
  options="--t.epochs 1000 --local.mode fd_neg"

elif [ "$2" = "neg-half-local" ]; then
  mode="glob local classifier"
  options="--t.epochs 1000 --local.mode fd_neg --local.scale 0.5"
fi


# Execute command
${cmd} ${mode} ${base_options} ${options}
