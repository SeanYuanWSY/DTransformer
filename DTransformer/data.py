import linecache
import math
import subprocess
import sys

import torch
from torch.utils.data import DataLoader


class Batch:
    def __init__(self, data, fields, seq_len=None):
        self.data = data
        self.fields = fields
        self.field_index = {f: i for i, f in enumerate(fields)}
        self.seq_len = seq_len

    def get(self, field):
        """获取单个字段的数据，返回张量"""
        if field not in self.field_index:
            return None

        tensor_data = self.data[self.field_index[field]]

        # 如果没有seq_len，直接返回
        if self.seq_len is None:
            return tensor_data

        # 如果有seq_len，需要分割序列
        L = tensor_data.shape[1]
        chunks = []
        for i in range(math.ceil(L / self.seq_len)):
            chunk = tensor_data[:, i * self.seq_len: (i + 1) * self.seq_len]
            chunks.append(chunk)
        return chunks  # 返回分块列表


class KTData:
    def __init__(
            self,
            data_path,
            inputs,
            batch_size=1,
            seq_len=None,
            shuffle=False,
            num_workers=0,
    ):
        self.data = Lines(data_path, group=len(inputs) + 1)
        self.loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=transform_batch,
            num_workers=num_workers,
        )
        self.inputs = inputs
        self.seq_len = seq_len

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """返回单个样本的Batch对象"""
        # 读取数据行，跳过第一行（可能是header）
        lines = self.data[index][1:]

        # 将每一行转换为整数列表（不使用torch.tensor，因为长度可能不同）
        parsed_data = []
        for line in lines:
            try:
                values = [int(x) for x in line.strip().split(",")]
                parsed_data.append(values)
            except ValueError as e:
                print(f"Error parsing line: {line}")
                raise e

        # 转换为张量列表（每个字段一个张量）
        tensors = []
        for field_data in parsed_data:
            tensors.append(torch.tensor(field_data, dtype=torch.long))

        return Batch(tensors, self.inputs, self.seq_len)


def transform_batch(batch):
    """
    合并多个样本为一个batch
    batch: list of Batch objects
    """
    if len(batch) == 0:
        return None

    # 获取配置
    fields = batch[0].fields
    seq_len = batch[0].seq_len

    # 收集每个字段的所有序列
    num_fields = len(fields)
    field_sequences = [[] for _ in range(num_fields)]

    for b in batch:
        for i, tensor in enumerate(b.data):
            field_sequences[i].append(tensor)

    # 对每个字段的序列进行填充
    padded_tensors = []
    for sequences in field_sequences:
        # 使用 pad_sequence 填充到相同长度
        padded = torch.nn.utils.rnn.pad_sequence(
            sequences,
            batch_first=True,
            padding_value=-1,
        )
        padded_tensors.append(padded)

    return Batch(padded_tensors, fields, seq_len)


class Lines:
    def __init__(self, filename, skip=0, group=1, preserve_newline=False):
        self.filename = filename
        with open(filename):
            pass
        if sys.platform == "win32":
            linecount = sum(1 for _ in open(filename))
        else:
            output = subprocess.check_output(("wc -l " + filename).split())
            linecount = int(output.split()[0])
        self.length = (linecount - skip) // group
        self.skip = skip
        self.group = group
        self.preserve_newline = preserve_newline

    def __len__(self):
        return self.length

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, item):
        d = self.skip + 1
        if isinstance(item, int):
            if item < len(self):
                if self.group == 1:
                    line = linecache.getline(self.filename, item + d)
                    if not self.preserve_newline:
                        line = line.strip("\r\n")
                else:
                    line = [
                        linecache.getline(self.filename, d + item * self.group + k)
                        for k in range(self.group)
                    ]
                    if not self.preserve_newline:
                        line = [l.strip("\r\n") for l in line]
                return line

        elif isinstance(item, slice):
            low = 0 if item.start is None else item.start
            low = _clip(low, -len(self), len(self) - 1)
            if low < 0:
                low += len(self)
            high = len(self) if item.stop is None else item.stop
            high = _clip(high, -len(self), len(self))
            if high < 0:
                high += len(self)
            ls = []
            for i in range(low, high):
                if self.group == 1:
                    line = linecache.getline(self.filename, i + d)
                    if not self.preserve_newline:
                        line = line.strip("\r\n")
                else:
                    line = [
                        linecache.getline(self.filename, d + i * self.group + k)
                        for k in range(self.group)
                    ]
                    if not self.preserve_newline:
                        line = [l.strip("\r\n") for l in line]
                ls.append(line)

            return ls

        raise IndexError


def _clip(v, low, high):
    if v < low:
        v = low
    if v > high:
        v = high
    return v