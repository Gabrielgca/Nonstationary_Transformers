# ps aux | grep python
# pkill -f "python -u run.py --is_training 1"
# Before
# train 34369
# val 11425
# test 11425
# After
# train 48585
# val 10165
# test 10165
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm2.csv \
#   --model_id ETTm2_96_96 \
#   --model ns_Transformer \
#   --data ETTm2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --gpu 0 \
#   --des 'Exp_h256_l2' \
#   --p_hidden_dims 256 256 \
#   --p_hidden_layers 2 \
#   --itr 3 

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm2.csv \
#   --model_id ETTm2_96_192 \
#   --model ns_Transformer \
#   --data ETTm2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --gpu 0 \
#   --des 'Exp_h64_l2' \
#   --p_hidden_dims 64 64 \
#   --p_hidden_layers 2 \
#   --itr 3  

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm2.csv \
#   --model_id ETTm2_96_336 \
#   --model ns_Transformer \
#   --data ETTm2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --gpu 0 \
#   --des 'Exp_h256_l2' \
#   --p_hidden_dims 256 256 \
#   --p_hidden_layers 2 \
#   --itr 3  

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm2.csv \
#   --model_id ETTm2_96_720 \
#   --model ns_Transformer \
#   --data ETTm2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --gpu 0 \
#   --des 'Exp_h256_l2' \
#   --p_hidden_dims 256 256 \
#   --p_hidden_layers 2 \
#   --itr 3  

#!/bin/bash

hidden_list=(256)
pred_list=(96 192 336 720)
ext_list=(1 0)  # 1 to extend with diff, 0 to not extend with diff
features_list=(M S)

for ext in "${ext_list[@]}"; do
  for h in "${hidden_list[@]}"; do
    des="Exp_h${h}_l2"

    for p in "${pred_list[@]}"; do
      for feat in "${features_list[@]}"; do

        # Set enc_in, dec_in, c_out based on feature type
        if [ "$feat" = "S" ]; then
          enc=1
          dec=1
          cout=1
        else
          enc=7
          dec=7
          cout=7
        fi

        echo "Running feat=${feat}, enc_in=${enc}, pred_len=${p}, des=${des}, extend_mlp=${ext}"

        python -u run.py \
          --is_training 1 \
          --root_path ./dataset/ETT-small/ \
          --data_path ETTm2_abrupt.csv \
          --model_id ETTm2_96_${p} \
          --model ns_Transformer \
          --data ETTm2 \
          --features ${feat} \
          --seq_len 96 \
          --label_len 48 \
          --pred_len ${p} \
          --e_layers 2 \
          --d_layers 1 \
          --extend_mlp ${ext} \
          --enc_in ${enc} \
          --dec_in ${dec} \
          --c_out ${cout} \
          --gpu 0 \
          --des "${des}" \
          --p_hidden_dims ${h} ${h} \
          --p_hidden_layers 2 \
          --itr 3

      done
    done
  done
done

