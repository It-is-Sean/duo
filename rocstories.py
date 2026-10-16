"""Shared paper-packed ROCStories cache and optimizer-step experiment planning.

The cache is built from raw documents by LangFlow's training/data.py. Reading
the same binary blocks avoids re-tokenizing/truncating stories differently in
the two experiments. No runtime import of LangFlow is required.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler

PROTOCOL = 'paper-packed-v1'


def read_metadata(cache_dir):
  metadata = json.loads((Path(cache_dir) / 'metadata.json').read_text())
  if metadata.get('fingerprint', {}).get('protocol') != PROTOCOL:
    raise ValueError('Expected paper-packed-v1 cache built from raw documents')
  return metadata


class PackedROCStories(Dataset):
  def __init__(self, cache_dir, split, block_size, tokenizer=None):
    self.metadata = read_metadata(cache_dir)
    self.fingerprint = self.metadata['fingerprint']
    fp = self.fingerprint
    if fp['sequence_length'] != block_size:
      raise ValueError('Packed cache length does not match model.length')
    if tokenizer is not None:
      # Duo adds a PAD vocabulary entry; it never occurs in packed blocks.
      if (tokenizer.vocab_size != fp['vocab_size']
          or tokenizer.bos_token_id != fp['bos_token_id']
          or tokenizer.eos_token_id != fp['eos_token_id']):
        raise ValueError('Packed cache tokenizer does not match Duo tokenizer')
    self.count = self.metadata['splits'][split]['blocks']
    self.block_size = block_size
    self.path = Path(cache_dir) / f'{split}.bin'
    self.dtype = np.dtype(fp['dtype'])
    if self.dtype not in (np.dtype('uint16'), np.dtype('uint32')):
      raise ValueError('Invalid packed token dtype')
    if self.count < 1 or self.path.stat().st_size != self.count * block_size * self.dtype.itemsize:
      raise ValueError('Incomplete packed token cache')
    self.tokens = None

  def __len__(self):
    return self.count

  def __getitem__(self, index):
    if not 0 <= index < self.count:
      raise IndexError(index)
    if self.tokens is None:
      self.tokens = np.memmap(self.path, mode='r', dtype=self.dtype,
                             shape=(self.count, self.block_size))
    ids = torch.from_numpy(np.array(self.tokens[index], dtype=np.int64, copy=True))
    return {'input_ids': ids, 'attention_mask': torch.ones(self.block_size)}

  def __getstate__(self):
    return {**self.__dict__, 'tokens': None}


class FullGlobalBatchSampler(DistributedSampler):
  """Drop only the shuffled epoch tail that cannot fill a global update.

    A DataLoader drop_last alone is insufficient with gradient accumulation:
    its final microbatches may still form an underweighted partial update.
  """
  def __init__(self, dataset, global_batch_size, **kwargs):
    super().__init__(dataset, drop_last=True, **kwargs)
    if global_batch_size % self.num_replicas:
      raise ValueError('Global batch must be divisible by world size')
    self.total_size = len(dataset) // global_batch_size * global_batch_size
    self.num_samples = self.total_size // self.num_replicas
    if not self.num_samples:
      raise ValueError('Dataset cannot fill one global batch')
    self.start_index = 0

  def __iter__(self):
    indices = list(super().__iter__())
    start = self.start_index
    self.start_index = 0
    return iter(indices[start:])


def experiment_plan(blocks, epochs, global_batch, micro_batch, world_size,
                    warmup_steps=400, curriculum_fraction=0.5):
  if min(blocks, epochs, global_batch, micro_batch, world_size) < 1:
    raise ValueError('Dataset, epoch and batch counts must be positive')
  if global_batch % (micro_batch * world_size):
    raise ValueError('Global batch must divide evenly across GPUs and accumulation')
  updates = blocks // global_batch
  total = epochs * updates
  if updates < 2 or not 0 <= warmup_steps < total:
    raise ValueError('Run must contain full updates and extend past LR warmup')
  if not 0 <= curriculum_fraction < 1:
    raise ValueError('Curriculum must leave a discrete-training phase')
  accumulation = global_batch // (micro_batch * world_size)
  checkpoint = max(1, updates // 2)
  return dict(updates_per_epoch=updates, max_steps=total,
              curriculum_end=int(total * curriculum_fraction),
              checkpoint_every=checkpoint,
              validation_microbatches=checkpoint * accumulation,
              accumulation=accumulation,
              dropped_blocks_per_epoch=blocks - updates * global_batch)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--cache-dir', type=Path, required=True)
  parser.add_argument('--epochs', type=int, default=20)
  parser.add_argument('--global-batch', type=int, default=128)
  parser.add_argument('--micro-batch', type=int, default=4)
  parser.add_argument('--world-size', type=int, default=2)
  parser.add_argument('--warmup-steps', type=int, default=400)
  parser.add_argument('--curriculum-fraction', type=float, default=0.5)
  args = parser.parse_args()
  plan = experiment_plan(read_metadata(args.cache_dir)['splits']['train']['blocks'],
                         args.epochs, args.global_batch, args.micro_batch,
                         args.world_size, args.warmup_steps, args.curriculum_fraction)
  print(' '.join(str(plan[k]) for k in ('max_steps', 'curriculum_end',
                                      'checkpoint_every', 'validation_microbatches')))
