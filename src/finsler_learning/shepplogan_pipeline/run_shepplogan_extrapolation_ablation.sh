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

base_export_path="/home/tblanchard/phd/finsler_learning/results/shepplogan/runs_2026-11-11_extrapolation_ablation"
base_trained_path="/home/tblanchard/phd/finsler_learning/results/shepplogan/runs_2026-11-11"


latent_dim=64
lbda=1.0
beta=1.0
b_size=128
b_size_eval=16
wd_vae=0
n_epochs_vae=1000
n_epochs_randers=50
n_epochs_regressor=500
n_pts=25
omega_strength=0.9

eta_list=(0.999 0.99 0.9 0.7 0.5)
# ema_list=(0.9 0.8 0.6 0.4 0)
ema_list=(1.0)
# alpha_list=(5.0 3.0 1.0 0.5 0.1)
alpha_list=(5.0)

use_symmetric_framework=1
for eta in "${eta_list[@]}"; do
    for ema_moment in "${ema_list[@]}"; do
        for alpha in "${alpha_list[@]}"; do
            echo "==================================================="
            echo "==================================================="
            echo "Running eta $eta, ema_moment $ema_moment, alpha $alpha"
            echo "==================================================="
            echo "==================================================="
            start_time_current=$(date +%s)

            export_path=$base_export_path/eta_$eta/ema_moment_$ema_moment/alpha_$alpha
            # Copy the trained model to the new export path
            mkdir -p $export_path
            cp -r $base_trained_path/* $export_path/
            rm -r $export_path/extrapolation_*
            rm -r $export_path/geodesic_evaluation_*

            
            if [[ "$use_symmetric_framework" -eq 1 ]]; then
                echo "Running symmetric framework for eta $eta, ema_moment $ema_moment, alpha $alpha"
                uv run shepplogan-pipeline \
                    --latent_dim $latent_dim \
                    --lbda $lbda \
                    --beta $beta \
                    --omega_strength $omega_strength \
                    --export_path $export_path \
                    --b_size_evaluation $b_size_eval \
                    --wd_vae $wd_vae \
                    --skip_vae \
                    --n_epochs_vae $n_epochs_vae \
                    --skip_randers \
                    --n_epochs_randers $n_epochs_randers \
                    --skip_regressor \
                    --n_epochs_regressor $n_epochs_regressor \
                    --b_size $b_size \
                    --eta $eta \
                    --alpha $alpha \
                    --ema_moment $ema_moment \
                    --n_pts $n_pts \
                    --eval_extrapolation \
                    --eval_quality_extrapolation
            else
                echo "Running non-symmetric framework only for eta $eta, ema_moment $ema_moment, alpha $alpha"
                uv run shepplogan-pipeline \
                    --latent_dim $latent_dim \
                    --lbda $lbda \
                    --beta $beta \
                    --omega_strength $omega_strength \
                    --export_path $export_path \
                    --b_size_evaluation $b_size_eval \
                    --wd_vae $wd_vae \
                    --skip_vae \
                    --n_epochs_vae $n_epochs_vae \
                    --skip_randers \
                    --n_epochs_randers $n_epochs_randers \
                    --skip_regressor \
                    --n_epochs_regressor $n_epochs_regressor \
                    --b_size $b_size \
                    --eta $eta \
                    --alpha $alpha \
                    --ema_moment $ema_moment \
                    --n_pts $n_pts \
                    --eval_extrapolation \
                    --eval_quality_extrapolation \
                    --no_symmetric
            fi
            end_time_current=$(date +%s)
            elapsed_time_current=$((end_time_current - start_time_current))
            hours_current=$((elapsed_time_current / 3600))
            minutes_current=$(( (elapsed_time_current % 3600) / 60 ))
            seconds_current=$((elapsed_time_current % 60))
            printf "Elapsed time for eta $eta, ema_moment $ema_moment, alpha $alpha: %02d:%02d:%02d (hh:mm:ss)\n" $hours_current $minutes_current $seconds_current
        done
    done
    use_symmetric_framework=0
done


# End time
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
hours=$((elapsed_time / 3600))
minutes=$(( (elapsed_time % 3600) / 60 ))
seconds=$((elapsed_time % 60))

printf "Total elapsed time: %02d:%02d:%02d (hh:mm:ss)\n" $hours $minutes $seconds

