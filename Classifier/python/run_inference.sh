## input file
dataset_path="/home/ykhsieh/CV/final/dataset/"
classifier_checkpoint="/home/ykhsieh/CV/final/CV_Final_Classifier/log-20230607123555/checkpoints/model_best_9997.pth"

## output file
output_path="/home/ykhsieh/CV/final/solution"

bin="python3 inference.py "
CUDA_VISIBLE_DEVICES=0 $bin \
--dataset_path ${dataset_path} \
--classifier_checkpoint ${classifier_checkpoint} \
--output_path ${output_path} \