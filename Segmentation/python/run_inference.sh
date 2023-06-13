time=$(date '+%Y%m%d%H%M%S')

## input file
dataset_path="/home/ykhsieh/CV/final/dataset/"
segmatation_checkpoint="/home/ykhsieh/CV/final/segmentation/log-20230603034837/checkpoints/model_best_9849.pth"

## output file
output_path="/home/ykhsieh/CV/final/output9"

bin="python3 inference.py "
CUDA_VISIBLE_DEVICES=1 $bin \
--dataset_path ${dataset_path} \
--segmatation_checkpoint ${segmatation_checkpoint} \
--output_path ${output_path} \