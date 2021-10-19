import random
import numpy as np
import datasets
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler


class MultiTaskDataset(Dataset):
    def __init__(self, all_datasets, total_len):
        self.all_datasets = all_datasets
        self.total_len = total_len

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        return [self.all_datasets[task_id][batch_id] for task_id, batch_id in enumerate(idx)]


class MultiTaskBatchSampler(BatchSampler):
    def __init__(self, all_datasets, fit_task_id, each_task_batch_size, distributed=False, shuffle=True):
        self.all_datasets = all_datasets
        self.fit_task_id = fit_task_id
        self.each_task_batch_size = each_task_batch_size
        self.ft_dataset = self.all_datasets[self.fit_task_id]
        self.distributed = distributed
        self.shuffle = shuffle
        if self.distributed:
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()

        self.all_trunc_datasets = None
        self._trunc_datasets()
        self.batch_indices = np.array(self._gen_batch_indices())

    def _trunc_datasets(self):
        ft_dataset_len = len(self.ft_dataset)
        all_trunc_datasets = []
        for task_id, dataset in enumerate(self.all_datasets):
            all_trunc_datasets.append(dataset[:ft_dataset_len])
        self.all_trunc_datasets = all_trunc_datasets

    def _gen_batch_indices(self, shuffle=False):
        ft_batch_indicies = [list(range(i, i + self.each_task_batch_size))
                             for i in range(0, len(self.ft_dataset), self.each_task_batch_size)]

        for i, idx in enumerate(ft_batch_indicies[-1]):
            if idx >= len(self.ft_dataset):
                ft_batch_indicies[-1][i] -= len(self.ft_dataset)
        ft_num_batches = len(ft_batch_indicies)

        all_batch_indices = []
        for task_id, dataset in enumerate(self.all_trunc_datasets):
            if task_id == self.fit_task_id:
                all_batch_indices.append(ft_batch_indicies)
            else:
                dataset_len = len(dataset[0])
                batch_indices = []
                i = 0
                while ft_num_batches - len(batch_indices) > 0:
                    index = list(range(i, i + self.each_task_batch_size))
                    if i + self.each_task_batch_size > dataset_len:
                        for j, cur_idx in enumerate(index):
                            while index[j] >= dataset_len:
                                index[j] -= dataset_len
                        i = index[-1] + 1
                    else:
                        i += self.each_task_batch_size
                    batch_indices.append(index)

                if shuffle:
                    random.shuffle(batch_indices)
                all_batch_indices.append(batch_indices)

        return all_batch_indices

    def __iter__(self):
        batch_indices = np.array(self._gen_batch_indices(self.shuffle))  # (n_task, num_batches, each_task_bs)
        batch_indices = batch_indices.transpose(1, 2, 0)  # (num_batches, each_task_bs, n_task)
        for cur_indices in batch_indices:
            batch_samples = cur_indices.tolist()
            if self.distributed:
                batch_samples = batch_samples[self.rank::self.num_replicas]
            yield batch_samples

    def __len__(self):
        return self.batch_indices.shape[1]


def create_single_domain_dataset(model_name, task, domain, data_dir, tokenizer, max_seq_len, batch_size, use_gpu,
                                 distributed, split, local_rank, cache_dir, train_ratio=0.9, dev_ratio=0.1,
                                 num_workers=4, load_from_multi_domain=False):
    if load_from_multi_domain:
        if task == 'mnli':
            domain_id = datasets.mnli_domains_to_ids[domain]
        else:  # task == 'amazon_review'
            domain_id = datasets.amazon_review_domains_to_ids[domain]
        examples, encoded_inputs, all_datasets = datasets.create_multi_domain_dataset(
            model_name, task, data_dir, tokenizer, max_seq_len, split, local_rank, train_ratio, dev_ratio,
            cache_dir)
        examples, encoded_inputs, dataset = examples[domain_id], encoded_inputs[domain_id], all_datasets[domain_id]
    else:
        examples, encoded_inputs, dataset = datasets.create_single_domain_dataset(
            model_name, task, domain, data_dir, tokenizer, max_seq_len, split, local_rank, train_ratio, dev_ratio,
            cache_dir)
    dataset_loader = _create_dataset_loader(dataset, batch_size, use_gpu, distributed, split, num_workers)
    return examples, encoded_inputs, dataset, dataset_loader


def create_multi_domain_dataset(model_name, task, fit_domain_id, data_dir, tokenizer, max_seq_len, each_task_batch_size,
                                use_gpu, distributed, split, local_rank, cache_dir, train_ratio=0.9, dev_ratio=0.1,
                                num_workers=4):
    examples, encoded_inputs, all_datasets = datasets.create_multi_domain_dataset(
        model_name, task, data_dir, tokenizer, max_seq_len, split, local_rank, train_ratio, dev_ratio, cache_dir)

    if split == 'train':
        train_batch_size = each_task_batch_size * dist.get_world_size() if distributed else each_task_batch_size
        loader = _create_multi_task_dataset_loader(all_datasets, fit_domain_id, train_batch_size, use_gpu, distributed, num_workers)
    else:  # split in ['val', 'dev', 'test']
        loader = [_create_dataset_loader(dataset, each_task_batch_size, use_gpu, distributed, split, num_workers)
                  for dataset in all_datasets]
    return examples, encoded_inputs, all_datasets, loader


def _create_dataset_loader(dataset, batch_size, use_gpu, distributed, split, num_workers=4):
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True if split == 'train' else False)
    else:
        sampler = RandomSampler(dataset) if split == 'train' else SequentialSampler(dataset)

    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=use_gpu)
    return loader


def _create_multi_task_dataset_loader(all_datasets, fit_task_id, batch_size, use_gpu, distributed, num_workers=4):
    batch_sampler = MultiTaskBatchSampler(all_datasets, fit_task_id, batch_size, distributed)
    dataset = MultiTaskDataset(all_datasets, len(batch_sampler))
    loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=use_gpu)
    return loader
