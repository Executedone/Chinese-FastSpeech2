''' 
_*_ coding: utf-8 _*_
Date: 2021/3/12
Author: 
Intent:
'''

config = {"pretrain_model_dir": "./chinese_roberta_wwm_ext_pytorch/",
          "model_dir": "./model",
          "init_checkpoint": "",

          "do_train": True,

          "warm_ratio": 0.1,
          "max_len": 40,
          "train_file": "./data/text_prosody_train.json",
          "dev_file": "./data/text_prosody_dev.json",
          "train_epoch": 100,
          "train_batch_size": 128,
          "dev_batch_size": 256,
          "weight_decay": 0.01,
          "learning_rate": 5e-5,
          "accum_steps": 1,
          "print_steps": 20,
          "eval_steps": 50,
          "save_ckpt_steps": 400000,
          "max_grad": 1,
          }


if __name__ == "__main__":
    pass