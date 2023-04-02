''' 
_*_ coding: utf-8 _*_
Date: 2021/3/13
Author: 
Intent:
'''

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)

import torch
from model import ProsodyModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import ProsodyDataset, prosody_collate
from tensorboardX import SummaryWriter
from tqdm import tqdm
import logging, os, random
import numpy as np

logger = logging.getLogger(__name__)


class Prosody(object):
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ProsodyModel(config)
        self.model.to(self.device)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_params = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.config['weight_decay']},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_params, lr=self.config['learning_rate'])
        self.optimizer.zero_grad()

        self.Loss = CrossEntropyLoss(reduction='none')

        self.train_set = None
        self.dev_set = None
        self.num_training_steps = None
        self.num_warm_steps = None
        self.scheduler = None

    def train(self):
        self.train_set = ProsodyDataset(self.config, self.config['train_file'])
        self.dev_set = ProsodyDataset(self.config, self.config['dev_file'])
        self.num_training_steps = self.config['train_epoch'] * (len(self.train_set) // self.config['train_batch_size'])
        self.num_warm_steps = self.config['warm_ratio'] * self.num_training_steps
        self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.num_warm_steps, num_training_steps=self.num_training_steps
        )

        epochs = [i for i in range(self.config['train_epoch'])]
        global_step = 0
        start_epoch = 0
        best_auc = 0.0
        writer = SummaryWriter(self.config['model_dir'])

        if self.config['init_checkpoint']:
            print(f'loading init_checkpoint from `{self.config["init_checkpoint"]}`')
            ckpt_dict = torch.load(self.config['init_checkpoint'], map_location=self.device)
            self.model.load_state_dict(ckpt_dict['model'])
            self.optimizer.load_state_dict(ckpt_dict['optimizer'])
            self.scheduler.load_state_dict(ckpt_dict['schedular'])
            global_step = ckpt_dict['global_step']
            start_epoch = ckpt_dict['epoch']

        self.model.train()

        for epoch in tqdm(epochs[start_epoch:]):
            loader = DataLoader(self.train_set, batch_size=self.config['train_batch_size'],
                                shuffle=True, num_workers=4, collate_fn=prosody_collate)
            train_batch = 0.0
            train_loss = 0.0
            for batch in tqdm(loader):
                global_step += 1

                batch = _move_to_device(batch, self.device)
                inputs_ids, inputs_masks, tokens_type_ids = batch['inputs_ids'], \
                                                            batch['inputs_masks'], \
                                                            batch['tokens_type_ids']
                bsz = inputs_ids.size(0)
                labels, loss_masks = batch['labels'].flatten(), inputs_masks.flatten()
                logits = self.model(inputs_ids, inputs_masks, tokens_type_ids)
                loss = self.Loss(logits, labels)
                loss = torch.sum(loss * loss_masks) / torch.sum(loss_masks)

                if self.config['accum_steps'] > 1:
                    loss = loss / self.config['accum_steps']

                train_loss += loss.item() * bsz
                train_batch += bsz

                if global_step % self.config['print_steps'] == 0:
                    print(f'current loss is {round(train_loss/train_batch, 6)}'
                          f'at {global_step} step on {epoch} epoch...')
                    writer.add_scalar('train_loss', train_loss/train_batch, global_step)
                    train_batch = 0.0
                    train_loss = 0.0

                if global_step > 1000 and global_step % self.config['save_ckpt_steps'] == 0:
                    print(f'saving ckpt at {global_step} step on {epoch} epoch...')
                    ckpt_dict = {'model': self.model.state_dict(),
                                 'optimizer': self.optimizer.state_dict(),
                                 'schedular': self.scheduler.state_dict(),
                                 'global_step': global_step,
                                 'epoch': epoch}
                    torch.save(ckpt_dict, os.path.join(self.config['model_dir'], f'ckpt-temp.bin'))

                if global_step % self.config['eval_steps'] == 0:
                    acc = self.eval()
                    writer.add_scalar('acc', acc, global_step)
                    print(f'acc: {acc}')
                    if acc > best_auc:
                        print(f'from {best_auc} -> {acc}')
                        print('saving models...')
                        torch.save(self.model.state_dict(), os.path.join(self.config['model_dir'], 'best_model.pt'))
                        best_auc = acc

                loss.backward()

                if global_step % self.config['accum_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad'])
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

    def eval(self):
        self.model.eval()
        dev_loader = DataLoader(self.dev_set, batch_size=self.config['dev_batch_size'],
                                shuffle=False, num_workers=4, collate_fn=prosody_collate)
        correct, total = 0, 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dev_loader)):
                batch = _move_to_device(batch, self.device)
                labels = batch['labels'].flatten()
                inputs_ids, inputs_masks, tokens_type_ids = batch['inputs_ids'], \
                                                            batch['inputs_masks'], \
                                                            batch['tokens_type_ids']
                label_masks = inputs_masks.flatten()

                logits = self.model(inputs_ids, inputs_masks, tokens_type_ids)
                logits = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                preds = (preds == labels).to(torch.long)
                preds = preds * label_masks
                correct += torch.sum(preds).item()
                total += torch.sum(label_masks).item()
            acc = correct / total
        self.model.train()
        return acc


def _move_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinidtic = True
    random.seed(seed)
    np.random.seed(seed)

def main():
    from config import config
    seed = 2022
    setup_seed(seed)
    p_model = Prosody(config)
    if config['do_train']:
        print('-----------start training-----------')
        p_model.train()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%y-%m-%d %H:%M:%S')
    main()