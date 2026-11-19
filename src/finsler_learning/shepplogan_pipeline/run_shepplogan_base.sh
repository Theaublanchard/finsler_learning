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
extrapolation_script_path="/home/tblanchard/phd/finsler_learning/src/finsler_learning/shepplogan_pipeline/shepplogan_extrapolation.py"


base_export_path="/home/tblanchard/phd/finsler_learning/results/shepplogan/runs_2026-11-11"
latent_dim=64
lbda=1.0
beta=1.0
b_size=128
b_size_eval=32
wd_vae=0
n_epochs_vae=1000
n_epochs_randers=50
n_epochs_regressor=500
eta=0.999
alpha=5.0
ema_moment=0.9
n_pts=25

omega_strength=1.0

# uv run shepplogan-pipeline \
#     --latent_dim $latent_dim \
#     --lbda $lbda \
#     --beta $beta \
#     --omega_strength $omega_strength \
#     --export_path "$base_export_path" \
#     --b_size_evaluation $b_size_eval \
#     --wd_vae $wd_vae \
#     --n_epochs_vae $n_epochs_vae \
#     --n_epochs_randers $n_epochs_randers \
#     --n_epochs_regressor $n_epochs_regressor \
#     --b_size $b_size \
#     --eta $eta \
#     --alpha $alpha \
#     --ema_moment $ema_moment \
#     --n_pts $n_pts \
#     --skip_regressor \
#     --skip_vae \
#     --eval_quality_interpolation \
#     --eval_interpolation \
#     --skip_randers \
#     # --no_symmetric \
#     # --eval_quality_extrapolation \
#     # --eval_extrapolation \
#     # --eval_regressor \

omega_strength=0.9
uv run shepplogan-pipeline \
    --latent_dim $latent_dim \
    --lbda $lbda \
    --beta $beta \
    --omega_strength $omega_strength \
    --export_path "$base_export_path" \
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
    --skip_randers \
    --eval_quality_extrapolation \
    --eval_extrapolation \
    # --eval_quality_interpolation \
    # --eval_interpolation \
    # --no_symmetric \
    # --no_randers \
    # --eval_regressor \

# spherical_script_path="/home/tblanchard/phd/finsler_learning/src/finsler_learning/shepplogan_pipeline/shepplogan_spherical.py"
# wasserstein_script_path="/home/tblanchard/phd/finsler_learning/src/finsler_learning/shepplogan_pipeline/shepplogan_wasserstein.py"
# uv run $spherical_script_path \
#     --config_path "$base_export_path" \
#     --eval_interpolation \
#     --eval_extrapolation \
#     --eval_quality_interpolation \
#     --eval_quality_extrapolation \
#     --skip_vae \

# uv run $wasserstein_script_path \
#     --base_path "$base_export_path" \
#     --n_bins 10 \
#     --n_epochs 500 \
#     --lr 1e-3 \
#     --eval_quality

# End time
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
hours=$((elapsed_time / 3600))
minutes=$(( (elapsed_time % 3600) / 60 ))
seconds=$((elapsed_time % 60))

printf "Total elapsed time: %02d:%02d:%02d (hh:mm:ss)\n" $hours $minutes $seconds
