#!/bin/bash
# TQDM_DISABLE=1
# export TQDM_DISABLE

PROJECT_ROOT="/home/tblanchard/phd/finsler_learning/"
cd "$PROJECT_ROOT"

# Redirect all output to both terminal and log file
logFile="/home/tblanchard/phd/finsler_learning/src/finsler_learning/shepplogan_pipeline/logs.txt"
exec > >(tee -a $logFile)
exec 2>&1

start_time=$(date +%s)
script_path="/home/tblanchard/phd/finsler_learning/src/finsler_learning/shepplogan_pipeline/shepplogan_pipeline.py"

base_trained_path="/home/tblanchard/phd/finsler_learning/results/shepplogan/runs_2026-11-11"
base_export_path="/home/tblanchard/phd/finsler_learning/results/shepplogan/runs_2026-11-11_dummy_acc"


latent_dim=64
lbda=1.0
beta=1.0
b_size=128
b_size_eval=16
wd_vae=0
n_epochs_vae=1000
n_epochs_randers=50
n_epochs_regressor=500
eta=0.999
alpha=5.0
ema_moment=0.9
n_pts=25


mkdir -p $base_export_path
# Copy the trained model to the new export path
cp -r $base_trained_path/* $base_export_path/
# We only retrain the Randers

omega_strength=1.0
uv run shepplogan-pipeline \
    --latent_dim $latent_dim \
    --lbda $lbda \
    --beta $beta \
    --omega_strength $omega_strength \
    --export_path $base_export_path \
    --b_size_evaluation $b_size_eval \
    --wd_vae $wd_vae \
    --n_epochs_vae $n_epochs_vae \
    --n_epochs_randers $n_epochs_randers \
    --n_epochs_regressor $n_epochs_regressor \
    --b_size $b_size \
    --eta $eta \
    --alpha $alpha \
    --ema_moment $ema_moment \
    --n_pts $n_pts \
    --skip_regressor \
    --skip_vae \
    --eval_quality_interpolation \
    --eval_interpolation \
    --no_symmetric \
    --use_dummy_acc \
    # --skip_randers \
    # --eval_extrapolation \
    # --eval_regressor \

omega_strength=0.9
uv run shepplogan-pipeline \
    --latent_dim $latent_dim \
    --lbda $lbda \
    --beta $beta \
    --omega_strength $omega_strength \
    --export_path $base_export_path \
    --b_size_evaluation $b_size_eval \
    --wd_vae $wd_vae \
    --n_epochs_vae $n_epochs_vae \
    --n_epochs_randers $n_epochs_randers \
    --n_epochs_regressor $n_epochs_regressor \
    --b_size $b_size \
    --eta $eta \
    --alpha $alpha \
    --ema_moment $ema_moment \
    --n_pts $n_pts \
    --skip_regressor \
    --skip_vae \
    --eval_extrapolation \
    --no_symmetric \
    --use_dummy_acc \
    # --skip_randers \
    # --eval_quality_interpolation \
    # --eval_interpolation \
    # --eval_regressor \


# End time
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
hours=$((elapsed_time / 3600))
minutes=$(( (elapsed_time % 3600) / 60 ))
seconds=$((elapsed_time % 60))

printf "Total elapsed time: %02d:%02d:%02d (hh:mm:ss)\n" $hours $minutes $seconds

