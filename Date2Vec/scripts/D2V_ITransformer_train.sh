
for data_name in 'ETTh1'
do
for rate in 0.001 0.0005 0.0001 0.00005 0.00001
do
for pred in  48 96 336
do

python -u main.py \
    --model_name 'D2V_Fourier_ITransformer'\
    --train True \
    --resume False \
    --loss normal \
    --data_name $data_name \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred \
    --d_feature 7 \
    --c_out 7 \
    --features 'M' \
    --d_model 512 \
    --d_ff 1024 \
    --lr $rate \
    --batch_size 64 \
    --e_layers 2 \
    --d_layers 1 \
    --dropout 0.1\
    --patch_len 16\
    --stride 8\
    --n_heads 3\
    --stride 8\
    --T2V_outmodel 32\
    --info 'D2V_ITransformer ETTh1'\

done
done
done


for data_name in 'exchange'
do
for rate in 0.001 0.0005 0.0001 0.00005 0.00001
do
for pred in  48 96 336
do

python -u main.py \
    --model_name 'D2V_Fourier_ITransformer'\
    --train True \
    --resume False \
    --loss normal \
    --data_name $data_name \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred \
    --d_feature 8 \
    --c_out 8 \
    --features 'M' \
    --d_model 512 \
    --d_ff 1024 \
    --lr $rate \
    --batch_size 64 \
    --e_layers 2 \
    --d_layers 1 \
    --dropout 0.1\
    --patch_len 16\
    --stride 8\
    --n_heads 3\
    --stride 8\
    --T2V_outmodel 32\
    --info 'D2V_ITransformer exchange'\

done
done
done


for data_name in 'traffic'
do
for rate in 0.001 0.0005 0.0001 0.00005 0.00001
do
for pred in  48 96 336
do

python -u main.py \
    --model_name 'D2V_Fourier_ITransformer'\
    --train True \
    --resume False \
    --loss normal \
    --data_name $data_name \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred \
    --d_feature 862 \
    --c_out 862 \
    --features 'M' \
    --d_model 512 \
    --d_ff 1024 \
    --lr $rate \
    --batch_size 16 \
    --e_layers 2 \
    --d_layers 1 \
    --dropout 0.1\
    --patch_len 16\
    --stride 8\
    --n_heads 3\
    --stride 8\
    --T2V_outmodel 32\
    --info 'D2V_ITransformer traffic'\

done
done
done


for data_name in 'ETTh1'
do
for rate in 0.001 0.0005 0.0001 0.00005 0.00001
do
for pred in  48 96 336
do

python -u main.py \
    --model_name 'D2V_Fourier_PatchTST'\
    --train True \
    --resume False \
    --loss normal \
    --data_name $data_name \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred \
    --d_feature 7 \
    --c_out 7 \
    --features 'M' \
    --d_model 510 \
    --d_ff 1024 \
    --lr $rate \
    --batch_size 64 \
    --e_layers 2 \
    --d_layers 1 \
    --dropout 0.1\
    --patch_len 16\
    --stride 8\
    --n_heads 3\
    --stride 8\
    --T2V_outmodel 32\
    --info 'D2V_PatchTST ETTh1'\

done
done
done


for data_name in 'exchange'
do
for rate in 0.001 0.0005 0.0001 0.00005 0.00001
do
for pred in  48 96 336
do

python -u main.py \
    --model_name 'D2V_Fourier_PatchTST'\
    --train True \
    --resume False \
    --loss normal \
    --data_name $data_name \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred \
    --d_feature 8 \
    --c_out 8 \
    --features 'M' \
    --d_model 510 \
    --d_ff 1024 \
    --lr $rate \
    --batch_size 64 \
    --e_layers 2 \
    --d_layers 1 \
    --dropout 0.1\
    --patch_len 16\
    --stride 8\
    --n_heads 3\
    --stride 8\
    --T2V_outmodel 32\
    --info 'D2V_PatchTST exchange'\

done
done
done