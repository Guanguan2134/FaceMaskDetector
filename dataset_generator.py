import os, shutil
import glob
import random

data_dir1 = 'E:/Google 雲端硬碟/Python/untitled/ai/Final_project/Face-Mask-Detection-master/dataset'
with_dir = os.path.join(data_dir1, 'with_mask')
without_dir = os.path.join(data_dir1, 'without_mask')
with_mask = glob.glob(os.path.join(with_dir, '*.*'))
without_mask = glob.glob(os.path.join(without_dir, '*.*'))
data_dir2 = 'E:/Google 雲端硬碟/Python/untitled/ai/Final_project/Face-Mask-Detection-master/dataset1'
with_dir = os.path.join(data_dir2, 'with_mask')
without_dir = os.path.join(data_dir2, 'without_mask')
with_mask = with_mask + glob.glob(os.path.join(with_dir, '*.*'))
without_mask = without_mask + glob.glob(os.path.join(without_dir, '*.*'))
random.shuffle(with_mask)
random.shuffle(without_mask)

base_dir = 'E:/Google 雲端硬碟/Python/untitled/ai/Final_project/Face-Mask-Detection-master/datasets'
if not os.path.isdir(base_dir): os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
if not os.path.isdir(train_dir): os.mkdir(train_dir)

test_dir = os.path.join(base_dir, 'test')
if not os.path.isdir(test_dir): os.mkdir(test_dir)

def copyfiles(dataset, itemname, file, unum, dnum):
    path = os.path.join(dataset, itemname)
    if not os.path.isdir(path): os.mkdir(path)

    if os.listdir(path):
        shutil.rmtree(path)
        os.mkdir(path)

    for i in range(unum, dnum):
        fname = itemname + '.{}.jpg'.format(i)
        src = file[i]
        dst = os.path.join(path, fname)
        shutil.copyfile(src, dst)

def make_small(itemname, file):
    copyfiles(train_dir, itemname, file, 0, 1600)
    copyfiles(test_dir, itemname, file, 1600, len(file))

make_small('with_mask', with_mask)
make_small('without_mask', without_mask)
print('[INFO] Done.')






