
# searching
CUDA_VISIBLE_DEVICES=0 python search.py --model ENAS --scale 2 --patch_size 96 --save search_model --reset --data_test Set5 --layers 12 --init_channels 8 --entropy_coeff 1 --lr 0.001 --epoch 400 --flops_scale 0.2


# inferring
CUDA_VISIBLE_DEVICES=0 python derive.py --data_test Set5 --scale 2 --pre_train  ../experiment/search_model/model/model_best.pt  --test_only --self_ensemble --save_results --save result/ --train_controller False --model ENAS --layer 12 --init_channels 8 --seed 1


# re-training
CUDA_VISIBLE_DEVICES=0 python main.py --model arch --genotype HNAS_A --scale 2 --patch_size 96 --save retrain_result --reset --data_test Set5 --data_range 1-800/801-810 --layers 12 --init_channels 64 --lr 1e-3 --epoch 300 --upsampling_Pos 9 --n_GPUs 1


# testing
CUDA_VISIBLE_DEVICES=0 python main.py --data_test Set5+Set14+B100+Urban100+Manga109 --data_range 801-900 --scale 2 --pre_train  ../experiment/retrain_result/model/model_best.pt  --test_only --self_ensemble --save_results --save result_arch/ --train_controller False --model arch --genotype HNAS_A --layer 12 --init_channels 64 --upsampling_Pos 9

