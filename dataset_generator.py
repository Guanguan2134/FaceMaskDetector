from email.mime import base
import os, shutil
import glob
import random
import yaml
from tqdm import trange


with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

# Get the file name of mask image and shuffle
base_dir = 'datasets'
data_dir = os.path.join(base_dir, 'raw')
with_dir = os.path.join(data_dir, 'with_mask')
without_dir = os.path.join(data_dir, 'without_mask')
with_mask = glob.glob(os.path.join(with_dir, '*.*'))
without_mask = glob.glob(os.path.join(without_dir, '*.*'))
random.shuffle(with_mask)
random.shuffle(without_mask)
data_len = max(len(with_mask), len(without_mask))

print("[INFO] Making dataset folder...\r")
# Make data folder for training and testing
train_dir = os.path.join(base_dir, 'train')
if os.path.isdir(train_dir):
    shutil.rmtree(train_dir)
    os.mkdir(train_dir)
    os.mkdir(os.path.join(train_dir, "with_mask"))
    os.mkdir(os.path.join(train_dir, "without_mask"))
else:
    os.mkdir(train_dir)
    os.mkdir(os.path.join(train_dir, "with_mask"))
    os.mkdir(os.path.join(train_dir, "without_mask"))

test_dir = os.path.join(base_dir, 'test')
if os.path.isdir(test_dir):
    shutil.rmtree(test_dir)
    os.mkdir(test_dir)
    os.mkdir(os.path.join(test_dir, "with_mask"))
    os.mkdir(os.path.join(test_dir, "without_mask"))
else:
    os.mkdir(test_dir)
    os.mkdir(os.path.join(test_dir, "with_mask"))
    os.mkdir(os.path.join(test_dir, "without_mask"))
print("[INFO] Making dataset folder...done")

print("[INFO] Transfer file to training and testing dataset")
print("[INFO] Copy file to training dataset")
for i in trange(int(round(data_len*config['Data']['train_split_ratio']))):
    src = with_mask[i]
    dst = os.path.join(os.path.join(train_dir, "with_mask"), f"with_mask_{i}.jpg")
    shutil.copyfile(src, dst)

    src = without_mask[i]
    dst = os.path.join(os.path.join(train_dir, "without_mask"), f"without_mask_{i}.jpg")
    shutil.copyfile(src, dst)

print("[INFO] Copy file to testing dataset")
with_mask = with_mask[int(round(data_len*config['Data']['train_split_ratio'])):]
without_mask = without_mask[int(round(data_len*config['Data']['train_split_ratio'])):]
for i in trange(len(with_mask)):
    src = with_mask[i]
    dst = os.path.join(os.path.join(test_dir, "with_mask"), f"with_mask_{i}.jpg")
    shutil.copyfile(src, dst)

for i in trange(len(without_mask)):
    src = without_mask[i]
    dst = os.path.join(os.path.join(test_dir, "without_mask"), f"without_mask{i}.jpg")
    shutil.copyfile(src, dst)

print('[INFO] Done.')








