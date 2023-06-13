time=$(date '+%Y%m%d%H%M%S')

## input file
dataset_dir=$1
label_data="../../conf.json"

## output file
val_imgs_dir="../log-${time}/val_imgs"
learning_curv_dir="../log-${time}/curv"
check_point_root="../log-${time}/checkpoints"
log_root="../log-${time}"

batch_size=20
lr=0.001
num_epochs=200
m1=15
m2=40
m3=60
if [ ! -d "../train-${time}" ]; then
mkdir -p ../log-${time}/{checkpoints,python_backups,val_imgs,curv}
fi

cp trainer.py ../log-${time}/python_backups
cp module.py ../log-${time}/python_backups
cp data.py ../log-${time}/python_backups
cp loss.py ../log-${time}/python_backups
cp utils.py ../log-${time}/python_backups
cp run_train.sh ../log-${time}/python_backups 

train_bin="python3 train.py "
$train_bin \
--dataset_dir ${dataset_dir} \
--label_data ${label_data} \
--val_imgs_dir ${val_imgs_dir} \
--learning_curv_dir ${learning_curv_dir} \
--check_point_root ${check_point_root} \
--log_root ${log_root} \
--batch_size ${batch_size} \
--lr ${lr} \
--num_epochs ${num_epochs} \
--milestones ${m1} ${m2} ${m3} \
