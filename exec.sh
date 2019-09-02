#!/bin/bash

# $1 = docker or local
# $2 = device id
# $3 = name
# $4 = evaluation method (optional)

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
if [ "$4" = "classification" ]; then
  name="DIM_CIFAR10_cls_$3"
else
  name="DIM_CIFAR10_$3"
fi
base_options="--d.source CIFAR10 --t.no_ascii --device $2 -n ${name}"


# Other options
ckpt_path="$HOME/.cortex/ckpt"
if [ "$4" = "classification" ]; then
  mode="classifier"
  options="--eval --t.epochs 1000 -L $ckpt_path/DIM_CIFAR10_$3/binaries/DIM_CIFAR10_$3_final.t7"

elif [ "$3" = "default" ]; then
  mode="local classifier"
  options="--t.epochs 1000"

elif [ "$3" = "neg-local" ]; then
  mode="glob local classifier"
  options="--t.epochs 1000 --local.mode fd_neg"

elif [ "$3" = "neg-half-local" ]; then
  mode="glob local classifier"
  options="--t.epochs 1000 --local.mode fd_neg --local.scale 0.5"

elif [ "$2" = "glob-local" ]; then
  mode="glob local classifier"
  options="--t.epochs 1000"

elif [ "$2" = "glob-half-local" ]; then
  mode="glob local classifier"
  options="--t.epochs 1000 --local.scale 0.5"
fi


# Execute command
${cmd} ${mode} ${base_options} ${options}
