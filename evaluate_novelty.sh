python ./scripts/compute_metrics_matbench.py \
    --train_path data/train.csv \
    --test_path data/test.csv \
    --gen_path data/adam_bs_100_lr_0.0001_decay_0_clip_1_A_119_W_28_N_21_a_1_w_1_l_1_Nf_5_Kx_16_Kl_16_h0_256_l_16_H_16_k_64_m_32_e_32_drop_0.5_0.5/output_160_struct.csv \
    --output_path data/adam_bs_100_lr_0.0001_decay_0_clip_1_A_119_W_28_N_21_a_1_w_1_l_1_Nf_5_Kx_16_Kl_16_h0_256_l_16_H_16_k_64_m_32_e_32_drop_0.5_0.5/ \
    --label 160 \
    --num_io_process 40
