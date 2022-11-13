import json
import torch
import numpy as np

import hifigan
from model import FastSpeech2, ScheduledOptim


def get_model(restore_step, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)
    if train_config["path"]["ckpt_path"]:
        ckpt_path = train_config["path"]["ckpt_path"]
        ckpt = torch.load(ckpt_path, map_location=device)["model"]
        print(f'loading model from `{ckpt_path}`...')

        try:
            model.load_state_dict(ckpt)
            print(f'loading model successfully...')
        except:
            print(f'warning: can not loading model directly, set `strict=False` to load model...')
            state_dict = {}
            for k, v in ckpt.items():
                if train and "speaker_emb" in k:
                    print('loading speaker_emb failed, init speaker_emb...')
                    continue
                if not train and "speaker_emb" in k:
                    print('loading speaker_emb failed, only loading speaker_emb[:1, :] weights...')
                    v = v[:1, :]
                state_dict[k] = v
            model.load_state_dict(state_dict, strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, restore_step
        )
        if train_config["path"]["ckpt_path"]:
            try:
                scheduled_optim.load_state_dict(torch.load(ckpt_path, map_location=device)["optimizer"])
                print('loading optimizer params successed, train from last step...')
            except:
                print('loading optimizer params failed, train from the beginning...')
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar", map_location=device)
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar", map_location=device)
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
