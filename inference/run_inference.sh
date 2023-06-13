time=$(date '+%Y%m%d%H%M%S')

## input file
dataset_path=$1
classifier_checkpoint="../CV_Final_Checkpoints/model_best_9998.pth"
segmatation_checkpoint="../CV_Final_Checkpoints/model_best_9858.pth"

## output file
output_path=$2

bin="python3 inference.py "
$bin \
--dataset_path ${dataset_path} \
--classifier_checkpoint ${classifier_checkpoint} \
--segmatation_checkpoint ${segmatation_checkpoint} \
--output_path ${output_path} \