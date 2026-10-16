import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from rocstories import PROTOCOL, PackedROCStories, FullGlobalBatchSampler, experiment_plan


class PackedTests(unittest.TestCase):
  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.path = Path(self.temporary.name)
    self.metadata = {'fingerprint': dict(protocol=PROTOCOL, sequence_length=8,
                     vocab_size=32, bos_token_id=1, eos_token_id=2, dtype='uint16'),
                     'splits': {'train': {'blocks': 19}}}
    (self.path / 'metadata.json').write_text(json.dumps(self.metadata))
    self.tokens = np.tile([1, 10, 2, 11, 12, 13, 14, 2], (19, 1)).astype('uint16')
    self.tokens.tofile(self.path / 'train.bin')

  def test_packing_keeps_internal_eos_and_all_positions(self):
    dataset = PackedROCStories(self.path, 'train', 8)
    np.testing.assert_array_equal(dataset[0]['input_ids'], self.tokens[0])
    self.assertEqual(dataset[0]['attention_mask'].sum(), 8)
    self.assertEqual(len(dataset), 19)

  def test_reject_legacy_cache(self):
    self.metadata['fingerprint']['protocol'] = 'story-aligned'
    (self.path / 'metadata.json').write_text(json.dumps(self.metadata))
    with self.assertRaisesRegex(ValueError, 'paper-packed'):
      PackedROCStories(self.path, 'train', 8)

  def test_length_tokenizer_and_truncated_cache_guards(self):
    with self.assertRaisesRegex(ValueError, 'length'):
      PackedROCStories(self.path, 'train', 64)
    wrong = SimpleNamespace(vocab_size=33, bos_token_id=1, eos_token_id=2)
    with self.assertRaisesRegex(ValueError, 'tokenizer'):
      PackedROCStories(self.path, 'train', 8, wrong)
    self.tokens[:-1].tofile(self.path / 'train.bin')
    with self.assertRaisesRegex(ValueError, 'Incomplete'):
      PackedROCStories(self.path, 'train', 8)

  def test_full_updates_disjoint_ranks_no_duplicates(self):
    dataset = PackedROCStories(self.path, 'train', 8)
    samplers = [FullGlobalBatchSampler(dataset, 8, num_replicas=2, rank=r, seed=42)
                for r in range(2)]
    rows = [list(s) for s in samplers]
    self.assertEqual([len(row) for row in rows], [8, 8])
    self.assertFalse(set(rows[0]) & set(rows[1]))
    samplers[0].start_index = 4
    self.assertEqual(list(samplers[0]), rows[0][4:])
    samplers[0].set_epoch(1)
    self.assertNotEqual(list(samplers[0]), rows[0])

  def test_schedule_counts_optimizer_updates_and_microbatches(self):
    plan = experiment_plan(41310, 20, 128, 4, 2)
    self.assertEqual(plan['max_steps'], 6440)
    self.assertEqual(plan['curriculum_end'], 3220)
    self.assertEqual(plan['checkpoint_every'], 161)
    self.assertEqual(plan['validation_microbatches'], 2576)
    self.assertEqual(plan['dropped_blocks_per_epoch'], 94)
    other = experiment_plan(41310, 20, 128, 8, 2)
    self.assertEqual(other['max_steps'], 6440)
    self.assertEqual(other['validation_microbatches'], 1288)

  def test_schedule_rejects_impossible_small_runs(self):
    for arguments in [(128, 20, 128, 4, 2), (41310, 1, 128, 4, 2),
                      (41310, 20, 128, 3, 2)]:
      with self.assertRaises(ValueError):
        experiment_plan(*arguments)

  def test_curriculum_boundary_and_log_temperature(self):
    from algo import DUO, DUO_BASE
    import lightning as L
    model = DUO.__new__(DUO)
    L.LightningModule.__init__(model)
    model.curriculum_start, model.curriculum_end = 0, 10
    model.gumbel_tau_log10_start = model.gumbel_tau_log10_end = -3
    model._trainer = SimpleNamespace(global_step=9)
    self.assertEqual(model._compute_gumbel_tau_inverse(), 1000)
    model._trainer.global_step = 10
    with patch.object(DUO_BASE, 'nll', return_value='discrete'):
      self.assertEqual(model.nll(None, None, None, train_mode=True), 'discrete')
      model._trainer.global_step = 0
      self.assertEqual(model.nll(None, None, None, train_mode=False), 'discrete')

  def test_ema_is_update_adaptive_not_fixed_09999(self):
    from models.ema import ExponentialMovingAverage
    parameter = torch.nn.Parameter(torch.zeros(1))
    ema = ExponentialMovingAverage([parameter], 0.9999)
    with torch.no_grad():
      parameter.fill_(1)
    ema.update([parameter])
    self.assertAlmostEqual(ema.shadow_params[0].item(), 9 / 11, places=6)


if __name__ == '__main__':
  unittest.main()
