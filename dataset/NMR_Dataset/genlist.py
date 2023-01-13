# Create gen_*.lst in NMR dataset
# For novel category generalization exp
import os
import numpy as np
import math
cats = sorted(os.listdir('.'))

train_cats = ['02691156', '02958343', '03001627']
prefix = 'gen_'
np.random.seed(987)

PROP_TEST = 0.25

for cat in cats:
    if not os.path.isdir(cat):
        continue
    objs = os.listdir(cat)
    objs = sorted([x for x in objs if os.path.isdir(os.path.join(cat, x, 'image'))])

    train_lst_path = os.path.join(cat, prefix + 'train.lst')
    test_lst_path = os.path.join(cat, prefix + 'test.lst')
    val_lst_path = os.path.join(cat, prefix + 'val.lst')

    if cat in train_cats:
        txt = '\n'.join(objs)
        with open(train_lst_path, 'w') as f:
            f.write(txt)
        with open(val_lst_path, 'w') as f:
            f.write('')
        with open(test_lst_path, 'w') as f:
            f.write('')
    else:
        objs = np.random.permutation(objs).tolist()
        num_test_objs = int(math.ceil(len(objs) * PROP_TEST))

        test_objs = objs[:num_test_objs]
        val_objs = objs[num_test_objs:]

        test_txt = '\n'.join(test_objs)
        val_txt = '\n'.join(val_objs)

        with open(train_lst_path, 'w') as f:
            f.write('')
        with open(val_lst_path, 'w') as f:
            f.write(val_txt)
        with open(test_lst_path, 'w') as f:
            f.write(test_txt)
