#!/bin/bash

export LC_CTYPE=C.UTF-8
docker-compose -f ./docker/docker-compose-gpu.yml run --rm -T experiment python3 scripts/main.py classifier --eval --d.source CIFAR10 -n DIM_CIFAR10_cls --t.epochs 1000 -L .cortex/output/DIM_CIFAR10/binaries/DIM_CIFAR10_final.t7 --t.no_ascii
