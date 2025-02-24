import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from PIL import Image
from utils.file_utils import *
import torchvision.transforms.functional as F
from utils.det_utils import compute_bbox_from_mask

class SquarePad:
    def __call__(self, image):
        # remove black area
        x0, y0, x1, y1 = compute_bbox_from_mask(np.array(image.convert('L')))
        if y1>y0 and x1>x0:
            image = F.crop(image, y0, x0, y1-y0, x1-x0)

        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')


class DTDDataProvider:
    """
    The data is available from https://www.robots.ox.ac.uk/~vgg/data/dtd/
    """

    def __init__(self, save_path=None, train_batch_size=32, test_batch_size=200, valid_size=None, n_worker=32,
                 resize_scale=0.08, distort_color=None, image_size=224,
                 num_replicas=None, rank=None):
        norm_mean = [0.5329876098715876, 0.474260843249454, 0.42627281899380676]
        norm_std = [0.26549755708788914, 0.25473554309855373, 0.2631728035662832]

        valid_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=3),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

        valid_data = datasets.ImageFolder(os.path.join(save_path, 'valid'), valid_transform)

        self.test = torch.utils.data.DataLoader(
            valid_data, batch_size=test_batch_size, shuffle=False,
            pin_memory=True, num_workers=n_worker)


class DTD(Dataset):
    def __init__(self, root_dir='./', type="train", size=None, transform=None, image_size=224):
        print('DTD Dataset created')
        self.type = type
        norm_mean = [0.5329876098715876, 0.474260843249454, 0.42627281899380676]
        norm_std = [0.26549755708788914, 0.25473554309855373, 0.2631728035662832]

        if type == 'train':
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=3),
                transforms.CenterCrop(image_size),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ])
            valid_data = datasets.ImageFolder(dtd_root, transform)
            self.image_list = []
        else:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=3),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ])
            self.image_list = []

        self.real_size = len(self.image_list)
        self.size = size
        self.transform = transform

    def __len__(self):

        return self.real_size if self.size is None else self.size

    def get_data(self, index):
        sample = {}
        image = Image.open(self.image_list[index])
        # load instances
        instance = Image.open(self.instance_list[index])
        instance, label = self.decode_instance(instance, self.instance_list[index])  # get semantic map and instance map
        sample['image'] = image
        sample['im_name'] = self.image_list[index]
        sample['instance'] = instance
        sample['label'] = label
        sample['im_shape'] = np.array([image.size])

        return sample

    def __getitem__(self, index):
        sample = self.get_data(index)
        # transform
        if (self.transform is not None):
            sample = self.transform(sample)
            return sample
        else:
            return sample


class cifar100(Dataset):
    def __init__(self, root_dir=None, type="train", size=None, transform=None, image_size=224):
        print('cifar100 Dataset created')
        self.type = type
        norm_mean = [0.49139968, 0.48215827, 0.44653124]
        norm_std = [0.24703233, 0.24348505, 0.26158768]
        if not isinstance(image_size, list):
            image_size = [image_size, image_size]

        if type == 'train':
            transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=3),
                transforms.CenterCrop(image_size[0]),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ])
            valid_data = torchvision.datasets.CIFAR100(root=root_dir, train=True, download=True, transform=None)
            self.image_list = valid_data.data
            self.label_list = valid_data.targets
        else:
            transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=3),  # BICUBIC interpolation
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ])
            valid_data = torchvision.datasets.CIFAR100(root=root_dir, train=False, download=True, transform=None)
            self.image_list = valid_data.data
            self.label_list = valid_data.targets

        self.real_size = len(self.image_list)
        self.size = size
        self.transform = transform

    def __len__(self):

        return self.real_size if self.size is None else self.size

    def get_data(self, index):
        sample = {}
        if self.type == 'train':
            index = random.randint(0, self.real_size - 1)
        image = self.image_list[index]
        label = self.label_list[index]
        sample['image'] = Image.fromarray(image)
        sample['label'] = label

        return sample

    def __getitem__(self, index):
        sample = self.get_data(index)
        # transform
        if (self.transform is not None):
            sample['image'] = self.transform(sample['image'])
            return sample
        else:
            return sample


class common_cls(Dataset):
    def __init__(self, config_file=None, type="train", size=None, im_size=224, imagenet_normalize=True):
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        self.type = type

        if type == 'train':
            transform = transforms.Compose([
                transforms.RandomChoice([
                    SquarePad(),
                    transforms.RandomResizedCrop(im_size, scale=(0.55, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation=3),
                ]),
                # SquarePad(),
                transforms.Resize((im_size, im_size), interpolation=3),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.RandomApply(
                    [transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD), ],
                    p=1.0 if imagenet_normalize else 0.0
                ),
            ])

        else:
            transform = transforms.Compose([
                SquarePad(),
                transforms.Resize((im_size,im_size), interpolation=3),  # BICUBIC interpolation
                transforms.ToTensor(),
                transforms.RandomApply(
                    [transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD), ],
                    p=1.0 if imagenet_normalize else 0.0
                ),
            ])

        info = load_pickle(config_file)[type]
        self.image_list, self.label_list = list(info.keys()), list(info.values())

        self.real_size = len(self.image_list)
        self.size = size
        self.transform = transform
        print('common_cls', type, 'sample num:', len(self.image_list))

    def __len__(self):
        return self.real_size if self.size is None else self.size

    def get_data(self, index):
        sample = {}
        im_path = self.image_list[index]
        label = self.label_list[index]
        sample['image'] = Image.open(im_path).convert('RGB')
        sample['label'] = label
        sample['im_path'] = im_path
        return sample

    def __getitem__(self, index):
        while 1:
            try:
                if self.type == 'train':
                    index = random.randint(0, self.real_size - 1)
                sample = self.get_data(index)
                sample['image'] = self.transform(sample['image'])
                break
            except:
                index = random.randint(0, self.real_size - 1)
                print('dataloader except')
                continue
        return sample


class common_reg(Dataset):
    def __init__(self, config_file=None, type="train", size=None, im_size=224, imagenet_normalize=True):
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        self.type = type

        if type == 'train':
            transform = transforms.Compose([
                SquarePad(),
                # SquarePad(),
                transforms.Resize((im_size, im_size), interpolation=3),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.RandomApply(
                    [transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD), ],
                    p=1.0 if imagenet_normalize else 0.0
                ),
            ])

        else:
            transform = transforms.Compose([
                SquarePad(),
                transforms.Resize((im_size,im_size), interpolation=3),  # BICUBIC interpolation
                transforms.ToTensor(),
                transforms.RandomApply(
                    [transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD), ],
                    p=1.0 if imagenet_normalize else 0.0
                ),
            ])

        info = load_pickle(config_file)[type]
        self.image_list, self.label_list = list(info.keys()), list(info.values())

        self.real_size = len(self.image_list)
        self.size = size
        self.transform = transform
        print('common_cls', type, 'sample num:', len(self.image_list))

    def __len__(self):
        return self.real_size if self.size is None else self.size

    def get_data(self, index):
        sample = {}
        im_path = self.image_list[index]
        label = self.label_list[index]
        sample['image'] = Image.open(im_path).convert('RGB')
        sample['label'] = label
        sample['im_path'] = im_path
        return sample

    def __getitem__(self, index):
        # if self.type == 'train':
        #     index = random.randint(0, self.real_size - 1)
        # sample = self.get_data(index)
        # sample['image'] = self.transform(sample['image'])
        while 1:
            try:
                if self.type == 'train':
                    index = random.randint(0, self.real_size - 1)
                sample = self.get_data(index)
                sample['image'] = self.transform(sample['image'])
                break
            except:
                index = random.randint(0, self.real_size - 1)
                print('dataloader except')
                continue
        return sample


if __name__ == '__main__':
    machine_root = '/home/xuzhenbo/'
    root_dir = machine_root + 'dishes_diminish_trainval/'
    prefix = "diminish_proposal_train.pkl"
    train_dataset = xxx_dataset(root_dir, prefix=prefix, type="train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    for info in train_loader:
        b = 1
