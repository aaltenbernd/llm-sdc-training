import torch
import itertools

from torch.utils.data import IterableDataset, Dataset, get_worker_info


class BufferDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.inputs = []
        self.labels = []

    def add_batch(self, batch):
        input_ids, labels = batch
        self.inputs.append(input_ids.cpu())
        self.labels.append(labels.cpu())

    def __len__(self):
        return sum(x.size(0) for x in self.inputs)

    def __getitem__(self, idx):
        flat_inputs = torch.cat(self.inputs, dim=0)
        flat_labels = torch.cat(self.labels, dim=0)
        return flat_inputs[idx], flat_labels[idx]

    def reset(self):
        self.inputs = []
        self.labels = []


class PreprocessedDataset(Dataset):
    def __init__(self, data, tokenizer, batch_size, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.data))
        batch = [self.data[i] for i in range(start, end)]

        tokenized_examples = [self.tokenizer(
            example["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ) for example in batch]

        input_ids = torch.stack([item["input_ids"].squeeze(0) for item in tokenized_examples])
        attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in tokenized_examples])

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class PreprocessedIterableDataset(IterableDataset):
    def __init__(self, data, tokenizer, batch_size, max_length):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            iter_data = iter(self.data)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iter_data = itertools.islice(self.data, worker_id, None, num_workers)

        batch = []
        for example in iter_data:
            tokenized_example = self.tokenizer(
                example["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            batch.append(tokenized_example)

            if len(batch) == self.batch_size:
                yield self._format_batch(batch)
                batch = []

        if batch:
            yield self._format_batch(batch)

    def _format_batch(self, batch):
        input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
        attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in batch])

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size