#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Proyectos/Astro/gdaespec/Earth_like_Atmosphere/Retrieval Tests"
source /home/dasan/anaconda3/etc/profile.d/conda.sh
conda activate POSEIDON

mkdir -p campaign_5obs/logs
python campaign_run_contam_queue.py --nproc 12 --keep-going > campaign_5obs/logs/contam_queue_master.log 2>&1
