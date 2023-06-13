from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--dataset_dir", type=str)
parser.add_argument("--label_data", type=str)

parser.add_argument("--val_imgs_dir", type=str)
parser.add_argument("--learning_curv_dir", type=str)
parser.add_argument("--check_point_root", type=str)
parser.add_argument("--log_root", type=str)


parser.add_argument("--batch_size", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--num_epochs", type=int)
parser.add_argument('--milestones', nargs='+', type=int)

args = parser.parse_args()