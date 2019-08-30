#!/bin/bash
 
export LC_CTYPE=C.UTF-8
docker-compose -f ./docker/docker-compose-gpu.yml run --rm -T experiment python3 scripts/main.py local classifier --d.source CIFAR10 -n DIM_CIFAR10 --t.epochs 1000 --t.no_ascii
