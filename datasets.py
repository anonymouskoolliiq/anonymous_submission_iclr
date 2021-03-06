import csv, torchvision, numpy as np, random, os
from PIL import Image

from torch.utils.data import Sampler, Dataset, DataLoader, BatchSampler, SequentialSampler, RandomSampler, Subset
from torchvision import transforms, datasets
from collections import defaultdict


class PairBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_iterations=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iterations = num_iterations

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for k in range(len(self)):
            if self.num_iterations is None:
                offset = k*self.batch_size
                batch_indices = indices[offset:offset+self.batch_size]
            else:
                batch_indices = random.sample(range(len(self.dataset)),
                                              self.batch_size)

            pair_indices = []
            for idx in batch_indices:
                y = self.dataset.get_class(idx)
                pair_indices.append(random.choice(self.dataset.classwise_indices[y]))

            yield batch_indices + pair_indices

    def __len__(self):
        if self.num_iterations is None:
            return (len(self.dataset)+self.batch_size-1) // self.batch_size
        else:
            return self.num_iterations

class IterBatchSampler(Sampler):
    def __init__(self, dataset, num_iterations, batch_size):

        self.dataset = dataset
        self.num_iterations = num_iterations
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(self.num_iterations):
            indices = random.sample(range(len(self.dataset)),
                                    self.batch_size)
            yield indices

    def __len__(self):
        return self.num_iterations


def _load_image(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class DatasetWrapper(Dataset):
    # Additinoal attributes
    # - indices
    # - classwise_indices
    # - num_classes
    # - get_class

    def __init__(self, dataset, indices=None):
        self.base_dataset = dataset
        if indices is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = indices

        # torchvision 0.2.0 compatibility
        if torchvision.__version__.startswith('0.2'):
            if isinstance(self.base_dataset, datasets.ImageFolder):
                self.base_dataset.targets = [s[1] for s in self.base_dataset.imgs]
            else:
                if self.base_dataset.train:
                    self.base_dataset.targets = self.base_dataset.train_labels
                else:
                    self.base_dataset.targets = self.base_dataset.test_labels

        self.classwise_indices = defaultdict(list)
        for i in range(len(self)):
            y = self.base_dataset.targets[self.indices[i]]
            self.classwise_indices[y].append(i)
        self.num_classes = max(self.classwise_indices.keys())+1

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)

    def get_class(self, i):
        return self.base_dataset.targets[self.indices[i]]

    def reset(self):
        self.__init__(self.base_dataset, self.indices)

class DoubleDatasetWrapper(Dataset):
    # Additinoal attributes
    # - indices
    # - classwise_indices
    # - num_classes
    # - get_class

    def __init__(self, dataset1, dataset2, indices=None):
        self.base_dataset = dataset1
        self.raw_dataset = dataset2
        if indices is None:
            self.indices = list(range(len(dataset1)))
        else:
            self.indices = indices

        # torchvision 0.2.0 compatibility
        if torchvision.__version__.startswith('0.2'):
            if isinstance(self.base_dataset, datasets.ImageFolder):
                self.base_dataset.targets = [s[1] for s in self.base_dataset.imgs]
            else:
                if self.base_dataset.train:
                    self.base_dataset.targets = self.base_dataset.train_labels
                else:
                    self.base_dataset.targets = self.base_dataset.test_labels

        self.classwise_indices = defaultdict(list)
        for i in range(len(self)):
            y = self.base_dataset.targets[self.indices[i]]
            self.classwise_indices[y].append(i)
        self.num_classes = max(self.classwise_indices.keys())+1

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]], self.raw_dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)

    def get_class(self, i):
        return self.base_dataset.targets[self.indices[i]]

def load_dataset(name, root, sample='default', **kwargs):
    # Dataset
    if name in ['imagenet','tinyimagenet', 'CUB200', 'STANFORD120', 'MIT67']:
        # TODO
        if name == 'tinyimagenet':
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            transform_test = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            train_val_dataset_dir = os.path.join(root, "train")
            test_dataset_dir = os.path.join(root, "val")

            val_idx = np.load(os.path.join('splits', name + '_val_idx.npy'))
            if kwargs.get('num_samples_per_class', None) is None:
                train_idx = [i for i in range(100000) if i not in val_idx]
            else:
                train_idx = np.load(os.path.join('splits',
                                                 '{}_{}_train_idx.npy'.format(name, kwargs['num_samples_per_class'])))
            if 'sam' in kwargs:
                trainset = DoubleDatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train), datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_test), train_idx)
            else:
                trainset = DatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train), train_idx)

            valset   = DatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_test), val_idx)
            testset  = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))

        elif name == 'imagenet':
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            train_val_dataset_dir = os.path.join(root, "train")
            test_dataset_dir = os.path.join(root, "val")

            if kwargs.get('num_samples_per_class', None) is None:
                if 'sam' in kwargs:
                    trainset = DoubleDatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train), datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_test))
                else:
                    trainset = DatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train))
            else:
                train_idx = np.load(os.path.join('splits',
                                                 '{}_{}_train_idx.npy'.format(name, kwargs['num_samples_per_class'])))
                if 'sam' in kwargs:
                    trainset = DoubleDatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train), datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_test), train_idx)
                else:
                    trainset = DatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train), train_idx)

            valset   = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))
            testset  = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))
        else:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            train_val_dataset_dir = os.path.join(root, name, "train")
            test_dataset_dir = os.path.join(root, name, "test")

            if kwargs.get('num_samples_per_class', None) is None:

                if 'sam' in kwargs:
                    trainset = DoubleDatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train), datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_test))
                else:
                    trainset = DatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train))

            else:
                train_idx = np.load(os.path.join('splits',
                                                 '{}_{}_train_idx.npy'.format(name, kwargs['num_samples_per_class'])))
                trainset = DatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train), train_idx.astype(int))

            valset   = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))
            testset  = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))

    elif name.startswith('cifar'):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(), #CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        if name == 'cifar10':
            CIFAR = datasets.CIFAR10
        else:
            CIFAR = datasets.CIFAR100

        val_idx = np.load(os.path.join('splits', name + '_val_idx.npy'))
        if kwargs.get('num_samples_per_class', None) is None:
            train_idx = [i for i in range(50000) if i not in val_idx]
        else:
            train_idx = np.load(os.path.join('splits',
                                             '{}_{}_train_idx.npy'.format(name, kwargs['num_samples_per_class'])))

        if 'sam' in kwargs:
            trainset = DoubleDatasetWrapper(CIFAR(root, train=True, download=True, transform=transform_train), CIFAR(root, train=True, download=True, transform=transform_test), train_idx)
        else:
            trainset = DatasetWrapper(CIFAR(root, train=True,  download=True, transform=transform_train), train_idx)
        valset   = DatasetWrapper(CIFAR(root, train=True,  download=True, transform=transform_test),  val_idx)
        testset  = DatasetWrapper(CIFAR(root, train=False, download=True, transform=transform_test))
    else:
        raise Exception('Unknown dataset: {}'.format(name))

    # Sampler
    if sample == 'default':
        if 'num_iterations' in kwargs and kwargs['num_iterations'] > 0:
            get_train_sampler = lambda d: IterBatchSampler(d, kwargs['num_iterations'], kwargs['batch_size'])
        else:
            get_train_sampler = lambda d: BatchSampler(RandomSampler(d), kwargs['batch_size'], False)
        get_test_sampler  = lambda d: BatchSampler(SequentialSampler(d), kwargs['batch_size'], False)

    elif sample == 'pair':
        if 'num_iterations' in kwargs and kwargs['num_iterations'] > 0:
            get_train_sampler = lambda d: PairBatchSampler(d, kwargs['batch_size'], num_iterations=kwargs['num_iterations'])
        else:
            get_train_sampler = lambda d: PairBatchSampler(d, kwargs['batch_size'])
        get_test_sampler  = lambda d: BatchSampler(SequentialSampler(d), kwargs['batch_size'], False)

    else:
        raise Exception('Unknown sampling: {}'.format(sampling))

    trainloader = DataLoader(trainset, batch_sampler=get_train_sampler(trainset), num_workers=4)
    valloader   = DataLoader(valset,   batch_sampler=get_test_sampler(valset), num_workers=4)
    testloader  = DataLoader(testset,  batch_sampler=get_test_sampler(testset), num_workers=4)

    return trainloader, valloader, testloader

