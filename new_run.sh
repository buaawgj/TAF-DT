# Setup tests

python experiment.py --seed 123 \
    --env halfcheetah --dataset medium   \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  \
    --v_target  --use_mean_reduce --relabel_adv  \
    --rtg_no_q  --adv_scale 2.0  \
    --eta 5.0 --K 5 --grad_norm 15.0  \
    --fixed_timestep  \


# walker2d

python experiment.py --seed 123 \        
    --env walker2d --dataset medium   \
    --tar_dataset morph_medium  \
    --eta 3.5 --grad_norm 5.0 --K 5  \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  \
    --v_target  --use_mean_reduce --relabel_adv  \
    --rtg_no_q  --adv_scale 2.0 --command_state_normalization  \
    --adv_mean_reduce --proportion 0.8  \
    --ot_filter  --ot_proportion 0.8  \

python experiment.py --seed 123 \
    --env walker2d --dataset medium   \
    --tar_dataset kinematic_medium  \
    --eta 3.5 --grad_norm 5.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  \
    --v_target  --use_mean_reduce --relabel_adv  \
    --rtg_no_q  --adv_scale 2.0 --command_state_normalization  \
    --adv_mean_reduce --proportion 0.8  \
    --ot_filter --ot_proportion 0.8  \


# hopper 

python experiment.py --seed 123 \
    --env hopper --dataset medium   \
    --eta 1.0 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  \
    --v_target  --use_mean_reduce --relabel_adv  \
    --rtg_no_q  --adv_scale 2.0 --adv_mean_reduce  \

# halfcheetah

python experiment.py --seed 123 \
    --env halfcheetah --dataset medium --tar_dataset morph_medium \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  \
    --v_target  --use_mean_reduce --relabel_adv  \
    --rtg_no_q  --adv_scale 2.0  \
    --eta 5.0 --K 5 --grad_norm 15.0  \
    --fixed_timestep --command_state_normalization  \
    --proportion 0.0

python experiment.py --seed 123 \
    --env halfcheetah --dataset medium --tar_dataset morph_medium \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  \
    --v_target  --use_mean_reduce --relabel_adv  \
    --rtg_no_q  --adv_scale 2.0  \
    --eta 5.0 --K 5 --grad_norm 15.0  \
    --fixed_timestep --command_state_normalization  \
    --proportion 0.8

python experiment.py --seed 123 \
    --env halfcheetah --dataset medium --tar_dataset morph_medium \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  \
    --v_target  --use_mean_reduce --relabel_adv  \
    --rtg_no_q  --adv_scale 2.0  \
    --eta 5.0 --K 5 --grad_norm 15.0  \
    --fixed_timestep --command_state_normalization  \
    --proportion 1.0