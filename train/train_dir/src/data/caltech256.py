"""
Data operations, will be used in train.py and eval.py
"""
import os
from PIL import Image
import numpy as np
import shutil

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
from mindspore.dataset.transforms import c_transforms
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.vision.utils import Inter
from mindspore.dataset import GeneratorDataset

from ..data.augment.auto_augment import _pil_interp, rand_augment_transform
from ..data.augment.mixup import Mixup
from ..data.augment.random_erasing import RandomErasing


def dataset_split(data_url, num=15):
    print("Begin split dataset")
    for root, dirs, files in os.walk(os.path.join(data_url, "train")):
        if dirs:
            continue
        if not os.path.exists(os.path.realpath(os.path.join(root.replace('train', 'val')))):
            os.makedirs(os.path.realpath(os.path.join(root.replace('train', 'val'))))
        for file in files[:num]:
            shutil.copyfile(os.path.join(root, file), os.path.join(root.replace('train', 'val'), file))
            os.remove(os.path.join(root, file))
            if os.path.exists(os.path.join(root, file)):
                print(os.path.join(root, file), 'has exist!')
    print("End split dataset")


class Caltech256:
    """Caltech256 Define"""
    def __init__(self, args, training=True):
        if args.val_split:
            dataset_split(args.train_data_path)
            val_dir = os.path.join(args.train_data_path, "val")
            self.val_dataset = create_dataset(val_dir, training=False, args=args)
        print('Create train and evaluate dataset.')
        if training:
            train_dir = os.path.join(args.train_data_path, "train")
            data_size = 0
            for root, dirs, files in os.walk(train_dir):
                for file in files:
                    data_size += os.path.getsize(os.path.join(root, file))
            print(f"train set size: {data_size/1048576:.4f} M")
            self.train_dataset = create_dataset(train_dir, training=True, args=args)
        else:
            test_ir = os.path.join(args.test_data_path, args.test_name)
            self.test_dataset = create_dataset_test(test_ir, args=args)


class CaltechTest(object):
    def __init__(self, dataset_dir):
        super(CaltechTest, self).__init__()
        names = os.listdir(dataset_dir)
        names.sort(key=lambda x: int(x.split('.')[0]))
        self.data = []
        for name in names:
            file_path = os.path.join(dataset_dir, name)
            image = Image.open(file_path).convert("RGB")
            image = np.array(image).astype(np.float32)
            self.data.append(image)

    def __getitem__(self, index):
        img = self.data[index]

        return img

    def __len__(self):
        return len(self.data)


def create_dataset_test(dataset_dir, args):
    data_set = GeneratorDataset(source=CaltechTest(dataset_dir), column_names=["image"], shuffle=False)

    image_size = args.image_size
    # mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    # std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    mean = [0.5 * 255, 0.5 * 255, 0.5 * 255]
    std = [0.5 * 255, 0.5 * 255, 0.5 * 255]
    # test transform complete
    if args.crop:
        transform_img = [
            c_vision.Resize((int(256 / 224 * image_size), int(256 / 224 * image_size)), interpolation=Inter.BICUBIC),
            c_vision.CenterCrop(image_size),
            c_vision.Normalize(mean=mean, std=std),
            c_vision.HWC2CHW()
        ]
    else:
        transform_img = [
            c_vision.Resize((int(image_size), int(image_size)), interpolation=Inter.BICUBIC),
            c_vision.Normalize(mean=mean, std=std),
            c_vision.HWC2CHW()
        ]
    data_set = data_set.map(input_columns="image", operations=transform_img)
    # apply batch operations
    data_set = data_set.batch(256)
    return data_set


def create_dataset(dataset_dir, args, repeat_num=1, training=True):
    """
    create a train or eval imagenet2012 dataset for SwinTransformer
    Args:
        dataset_dir(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1

    Returns:
        dataset
    """
    device_num, rank_id = _get_rank_info()
    shuffle = bool(training)
    class_indexing = dict(zip([str(i) for i in range(1, args.num_classes+1)], [i for i in range(0, args.num_classes)]))
    if device_num == 1 or not training:
        data_set = ds.ImageFolderDataset(dataset_dir, num_parallel_workers=args.num_parallel_workers,
                                         shuffle=shuffle, decode=True, class_indexing=class_indexing)
    else:
        data_set = ds.ImageFolderDataset(dataset_dir, num_parallel_workers=args.num_parallel_workers, shuffle=shuffle,
                                         num_shards=device_num, shard_id=rank_id, decode=True,
                                         class_indexing=class_indexing)

    image_size = args.image_size

    # define map operations
    # BICUBIC: 3
    if training:
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        aa_params = dict(
            translate_const=int(image_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        interpolation = args.interpolation
        auto_augment = args.auto_augment
        assert auto_augment.startswith('rand')
        aa_params['interpolation'] = _pil_interp(interpolation)

        transform_img = [
            c_vision.Resize((int(256 / 224 * image_size), int(256 / 224 * image_size)), interpolation=Inter.BICUBIC),
            c_vision.CenterCrop(image_size),
            c_vision.RandomHorizontalFlip(prob=0.5),
            py_vision.ToPIL()
        ]
        # transform_img += [rand_augment_transform(auto_augment, aa_params)]
        transform_img += [
            py_vision.ToTensor(),
            py_vision.Normalize(mean=mean, std=std),
            RandomErasing(args.re_prob, mode=args.re_mode, max_count=args.re_count)
        ]
    else:
        # mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        # std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        mean = [0.5 * 255, 0.5 * 255, 0.5 * 255]
        std = [0.5 * 255, 0.5 * 255, 0.5 * 255]
        # test transform complete
        if args.crop:
            transform_img = [
                c_vision.Resize((int(256 / 224 * image_size), int(256 / 224 * image_size)),
                                interpolation=Inter.BICUBIC),
                c_vision.CenterCrop(image_size),
                c_vision.Normalize(mean=mean, std=std),
                c_vision.HWC2CHW()
            ]
        else:
            transform_img = [
                c_vision.Resize((int(image_size), int(image_size)), interpolation=Inter.BICUBIC),
                c_vision.Normalize(mean=mean, std=std),
                c_vision.HWC2CHW()
            ]

    transform_label = c_transforms.TypeCast(mstype.int32)

    data_set = data_set.map(input_columns="image", num_parallel_workers=args.num_parallel_workers,
                            operations=transform_img)
    data_set = data_set.map(input_columns="label", num_parallel_workers=args.num_parallel_workers,
                            operations=transform_label)
    if (args.mix_up > 0. or args.cutmix > 0.) and not training:
        # if use mixup and not training(False), one hot val data label
        one_hot = c_transforms.OneHot(num_classes=args.num_classes)
        data_set = data_set.map(input_columns="label", num_parallel_workers=args.num_parallel_workers,
                                operations=one_hot)
    # apply batch operations
    data_set = data_set.batch(args.batch_size, drop_remainder=True,
                              num_parallel_workers=args.num_parallel_workers)

    if (args.mix_up > 0. or args.cutmix > 0.) and training:
        mixup_fn = Mixup(
            mixup_alpha=args.mix_up, cutmix_alpha=args.cutmix, cutmix_minmax=None,
            prob=args.mixup_prob, switch_prob=args.switch_prob, mode=args.mixup_mode,
            label_smoothing=args.label_smoothing, num_classes=args.num_classes)

        data_set = data_set.map(operations=mixup_fn, input_columns=["image", "label"],
                                num_parallel_workers=args.num_parallel_workers)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)
    ds.config.set_prefetch_size(4)
    return data_set


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        from mindspore.communication.management import get_rank, get_group_size
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id
