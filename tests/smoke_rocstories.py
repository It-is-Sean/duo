"""CPU integration: real Duo objective, curriculum switch, checkpoint and sampler.

Uses cached GPT-2 tokenizers and the released integral table, but a tiny DiT.
All data and checkpoints are temporary; no running GPU experiment is touched.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lightning as L
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

import main  # register the project's OmegaConf resolvers
import algo
import dataloader
from rocstories import PROTOCOL


class ObservedBatches(Callback):
  def __init__(self):
    self.rows = []

  def on_train_batch_start(self, trainer, model, batch, batch_idx):
    self.rows.append((trainer.current_epoch, int(batch['input_ids'][0, 1])))


def main_test():
  torch.set_num_threads(2)
  with tempfile.TemporaryDirectory(prefix='duo-packed-smoke-') as directory:
    work = Path(directory)
    fp = dict(protocol=PROTOCOL, sequence_length=8, dtype='uint16',
              vocab_size=50257, bos_token_id=50256, eos_token_id=50256)
    metadata = {'fingerprint': fp, 'splits': {}}
    for split, count in [('train', 9), ('validation', 3)]:
      ids = np.tile([50256, 100, 300, 50256, 400, 500, 600, 50256], (count, 1)).astype('uint16')
      ids[:, 1] += np.arange(count, dtype='uint16')
      ids.tofile(work / f'{split}.bin')
      metadata['splits'][split] = {'blocks': count}
    (work / 'metadata.json').write_text(json.dumps(metadata))
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'configs')):
      cfg = compose(config_name='config', overrides=[
        'data=rocstories', 'algo=duo', f'data.cache_dir={work}', 'seed=42',
        'model.hidden_size=16', 'model.cond_dim=16', 'model.n_heads=2',
        'model.n_blocks=1', 'model.length=8', 'model.dropout=0',
        'loader.global_batch_size=4', 'loader.batch_size=1',
        'loader.eval_global_batch_size=1', 'loader.eval_batch_size=1',
        'loader.num_workers=0', 'loader.pin_memory=false',
        'trainer.devices=1', 'trainer.accelerator=cpu', 'trainer.max_steps=6',
        'trainer.precision=32-true', 'algo.curriculum.start=0', 'algo.curriculum.end=3',
        'algo.curriculum.gumbel_tau_log10_start=-3', 'algo.curriculum.gumbel_tau_log10_end=-3',
        f'algo.curriculum.integral_cache_path={ROOT}/integral/gpt2.pkl',
        'lr_scheduler.num_warmup_steps=1', 'eval.generate_samples=false',
        'eval.gen_ppl_eval_model_name_or_path=gpt2', 'wandb=null'])
    tokenizer = dataloader.get_tokenizer(cfg)

    def train(name, checkpoint=None):
      L.seed_everything(42)
      train_dl, val_dl = dataloader.get_dataloaders(cfg, tokenizer)
      model = algo.DUO(cfg, tokenizer)
      observed = ObservedBatches()
      saver = ModelCheckpoint(dirpath=work / name, filename='step-{step}',
                              auto_insert_metric_name=False, every_n_train_steps=1,
                              save_top_k=-1, save_last=True)
      trainer = L.Trainer(accelerator='cpu', devices=1, max_steps=6, max_epochs=3,
                          accumulate_grad_batches=4, precision='32-true', logger=False,
                          enable_progress_bar=False, enable_model_summary=False,
                          callbacks=[observed, saver], num_sanity_val_steps=0,
                          val_check_interval=4, check_val_every_n_epoch=None)
      trainer.fit(model, train_dl, val_dl, ckpt_path=checkpoint)
      assert trainer.global_step == 6
      return observed, model

    continuous, model = train('continuous')
    resumed, _ = train('resumed', str(work / 'continuous/step-3.ckpt'))
    assert resumed.rows == continuous.rows[12:], (resumed.rows, continuous.rows[12:])
    state = torch.load(work / 'continuous/step-3.ckpt', weights_only=False, map_location='cpu')
    assert state['rocstories_fingerprint'] == fp
    assert state['packed_progress'] == {'epoch': 1, 'batches': 4}
    state['rocstories_fingerprint'] = {'protocol': 'story-aligned'}
    try:
      model.on_load_checkpoint(state)
    except ValueError:
      pass
    else:
      raise AssertionError('Legacy checkpoint was accepted')
    model.eval()
    samples = model.restore_model_and_sample(num_steps=4)
    assert samples.shape == (1, 8)
    print('PASS: CPU training, curriculum boundary, mid-epoch resume sample order, protocol guard and fixed-block generation')


if __name__ == '__main__':
  main_test()
