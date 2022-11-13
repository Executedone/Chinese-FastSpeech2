import re, os, json
import torch
import yaml
import numpy as np
from pypinyin import pinyin, Style, load_phrases_dict
from transformers import BertTokenizer

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from text import text_to_sequence
from text_normalization import text_normalization, PUNCTRUATION, POLY_PHRASE, NUM_DICT
import jieba
from pypinyin_dict.phrase_pinyin_data import large_pinyin
from transformer.ProsodyModel import CharEmbedding


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'####### device: {device} #######')


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def read_phrase_dict(phra_path):
    with open(phra_path, 'r', encoding='utf8') as f:
        phra_dict = json.load(f)
    return phra_dict


def _get_pinyins(pinyin_list, lexicon):
    new_pinyins = []
    temp = ''
    for p in pinyin_list:
        if p in lexicon:
            if temp:
                new_pinyins.append(temp)
                temp = ''
            new_pinyins.append(p)
        else:
            temp += p
    if temp:
        new_pinyins.append(temp)
    return new_pinyins


def add_userword(words):
    for w in words:
        jieba.add_word(w)


def is_seg(word_list, phrase_phone_dict):
    for w in word_list:
        if len(w) <= 1:
            flag = False
            return flag

    flag = False
    for w in word_list:
        if w in phrase_phone_dict:
            flag = True
            break
    return flag


def word_segment(text, phrase_phone_dict, min_len=3):
    new_word_list = []
    word_list = jieba.lcut(text, HMM=True)
    for w in word_list:
        if w in phrase_phone_dict:
            new_word_list.append(w)
            continue
        if len(w) >= min_len:
            jieba.del_word(w)
            sub_w_list = jieba.lcut(w, HMM=True)
            flag = is_seg(sub_w_list, phrase_phone_dict)
            if flag:
                new_word_list.extend(sub_w_list)
            else:
                new_word_list.append(w)
            jieba.add_word(w)
        else:
            new_word_list.append(w)
    return new_word_list


def correct_pinyin_special(text_list, phrase_phone_dict):
    correct_pinyins = []
    specail_p = []
    for text in text_list:
        if text in phrase_phone_dict:
            tmp = phrase_phone_dict[text]
        else:
            tmp = [p[0] for p in pinyin(text, style=Style.TONE3, strict=False, neutral_tone_with_five=True)]

        if len(specail_p) > 0:
            if re.findall(r'\d', tmp[0])[0] == '4':
                specail_p[-1] = re.sub(r'\d', '2', specail_p[-1])
            else:
                specail_p[-1] = re.sub(r'\d', '4', specail_p[-1])
            correct_pinyins.extend(specail_p)
            specail_p = []

        if '一' in text:
            index = text.find('一')
            if 0 <= index < len(tmp) - 1 and text[index + 1] not in list(NUM_DICT.values()):
                if re.findall(r'\d', tmp[index + 1])[0] == '4':
                    tmp[index] = re.sub(r'\d', '2', tmp[index])
                else:
                    tmp[index] = re.sub(r'\d', '4', tmp[index])
            else:
                pass

        if '不' in text:
            index = text.find('不')
            if 0 <= index < len(tmp) - 1:
                if re.findall(r'\d', tmp[index + 1])[0] == '4':
                    tmp[index] = re.sub(r'\d', '2', tmp[index])
                else:
                    tmp[index] = re.sub(r'\d', '4', tmp[index])
            elif index == len(tmp) - 1:
                specail_p.extend(tmp)
                tmp = []
            else:
                pass

        correct_pinyins.extend(tmp)
    return correct_pinyins


def correct_pinyin_tone3(pinyin_list):
    if len(pinyin_list) >= 2:
        for i in range(1, len(pinyin_list)):
            try:
                if re.findall(r'\d', pinyin_list[i-1])[0] == '3' and re.findall(r'\d', pinyin_list[i])[0] == '3':
                    pinyin_list[i-1] = pinyin_list[i-1].replace('3', '2')
            except IndexError:
                pass
    return pinyin_list


def get_char_embeds(text, length, char_model, tokenizer):
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    input_masks = [1] * len(input_ids)
    type_ids = [0] * len(input_ids)
    input_ids = torch.LongTensor([input_ids]).to(device)
    input_masks = torch.LongTensor([input_masks]).to(device)
    type_ids = torch.LongTensor([type_ids]).to(device)

    with torch.no_grad():
        char_embeds = char_model(input_ids, input_masks, type_ids).squeeze(0).cpu()

    assert char_embeds.size(0) == len(length)

    expand_vecs = list()
    for vec, leng in zip(char_embeds, length):
        vec = vec.expand(leng, -1)
        expand_vecs.append(vec)
    expand_embeds = torch.cat(expand_vecs, 0)

    assert expand_embeds.size(0) == sum(length)

    return expand_embeds.numpy()


def preprocess_mandarin(text, preprocess_config, char_model, tokenizer):
    large_pinyin.load()
    load_phrases_dict(POLY_PHRASE)
    add_userword(list(POLY_PHRASE.keys()))

    phones = []
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    phrase_phone_dict = read_phrase_dict(preprocess_config["path"]["phrase_phone_path"])
    add_userword(list(phrase_phone_dict.keys()))

    text = text_normalization(text)
    text_list = word_segment(text, phrase_phone_dict, min_len=3)
    print(text_list)

    # 改进处理
    pinyins = correct_pinyin_special(text_list, phrase_phone_dict)  # 从biaobei字典中获取拼音
    pinyins = _get_pinyins(pinyins, lexicon)  # 从pypinyin中获取拼音
    pinyins = correct_pinyin_tone3(pinyins)  # 声调校正
    # print("pinyin: ", pinyins)

    length = []
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
            length.append(len(lexicon[p]))
        elif p in PUNCTRUATION:
            if p == '+':
                phones += lexicon['jia1']
                length.append(1)
            else:
                phones.append("sp")
                length.append(1)
        else:
            phones.append("sp")
            length.append(1)

    if phones[-1] != 'sp':
        phones.append('sp')
        length.append(1)
        text += '。'

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Pinyin Sequence: {}".format(pinyins))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(text_to_sequence(phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]))

    try:
        assert sum(length) == len(sequence)
        char_embeds = get_char_embeds(text, length, char_model, tokenizer)
    except Exception as e:
        print(f'--WARNING-- get char embedding error as `{e}` --WARNING--')
        char_embeds = None

    return np.array(sequence), char_embeds


def synthesize(model, configs, vocoder, batchs, chars_embeds, control_values, result_path):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for i, batch in enumerate(batchs):
        batch = to_device(batch, device)

        if chars_embeds[i] is not None:
            char_embeds = torch.from_numpy(chars_embeds[i]).float().to(device)
            print('**** shape of char_embeds **** ', char_embeds.shape)
        else:
            char_embeds = None
            print('++++ char_embeds is none... ++++ ')
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                char_vecs=char_embeds,
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )

            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                result_path,
            )


def synthesize_all(text_file, result_path):
    restore_step = 0
    pitch_control = 1.0
    energy_control = 1.0
    duration_control = 1.0

    # Read Config
    preprocess_config_path = "./config/AISHELL3/preprocess.yaml"
    model_config_path = "./config/AISHELL3/model.yaml"
    train_config_path = "./config/AISHELL3/train.yaml"
    preprocess_config = yaml.load(open(preprocess_config_path, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_config_path, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Result Path
    os.makedirs(result_path, exist_ok=True)

    # Get model
    model = get_model(restore_step, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Get char model, char tokenizer
    char_model = CharEmbedding(preprocess_config['path']['char_model_path'])
    char_model.to(device)
    char_model.load_state_dict(
            torch.load(
                    os.path.join(preprocess_config['path']['char_model_path'], 'best_model.pt'),
                    map_location=device)
    )
    char_model.eval()
    char_tokenizer = BertTokenizer.from_pretrained(preprocess_config['path']['char_model_path'])

    # Read file and Preprocess texts
    with open(text_file, 'r', encoding='utf8') as f:
        for line in f:
            sample = line.strip().split()
            ids, raw_texts = sample[0], ''.join(sample[1:])
            speakers = np.array([0])
            print(ids)
            # 整句合成
            phones_seq, char_embeds = preprocess_mandarin(raw_texts, preprocess_config, char_model, char_tokenizer)
            texts = np.array([phones_seq])
            char_embeds = np.array([char_embeds]) if char_embeds is not None else char_embeds
            text_lens = np.array([len(texts[0])])
            batchs = [([ids], [raw_texts], speakers, texts, text_lens, max(text_lens))]
            chars_embeds = [char_embeds]

            control_values = pitch_control, energy_control, duration_control
            synthesize(model, configs, vocoder, batchs, chars_embeds, control_values, result_path)


# 构建音频合成类，为服务做准备
class SpeechSynthesis(object):
    def __init__(self, config_dir):
        print("loading built-in configs...")
        self.internal_conf, \
        self.preprocess_config, \
        self.model_config, \
        self.train_config = self._read_internal_config(config_dir)

        print("loading acoustic model...")
        self.model = get_model(0, self.internal_conf, device, train=False)
        print("loading vocoder...")
        self.vocoder = get_vocoder(self.model_config, device)
        print("loading prosody model...")
        self.char_model = CharEmbedding(self.preprocess_config['path']['char_model_path'])
        self.char_model.to(device)
        self.char_model.load_state_dict(
                torch.load(
                        os.path.join(self.preprocess_config['path']['char_model_path'], 'best_model.pt'),
                        map_location=device
                ),
                strict=False
        )
        self.char_model.eval()
        self.char_tokenizer = BertTokenizer.from_pretrained(self.preprocess_config['path']['char_model_path'])

        self.result_path = './'

        jieba.initialize()

    def _read_internal_config(self, config_dir):
        preprocess_config_path = os.path.join(config_dir, "preprocess.yaml")
        model_config_path = os.path.join(config_dir, "model.yaml")
        train_config_path = os.path.join(config_dir, "train.yaml")
        preprocess_config = yaml.load(open(preprocess_config_path, "r"), Loader=yaml.FullLoader)
        model_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)
        train_config = yaml.load(open(train_config_path, "r"), Loader=yaml.FullLoader)
        configs = (preprocess_config, model_config, train_config)
        return configs, preprocess_config, model_config, train_config

    def text2speech(self, text, pitch_control=1.0, energy_control=1.0, duration_control=1.0):
        if len(text) < 1:
            print('no texts!')
            return None
        else:
            raw_texts = re.sub(r"\s+", "", text.strip())
            speakers = np.array([0])

            # 整句合成
            print('starting text processing')
            phones_seq, char_embeds = preprocess_mandarin(raw_texts,
                                                          self.preprocess_config,
                                                          self.char_model,
                                                          self.char_tokenizer)
            texts = np.array([phones_seq])
            char_embeds = np.array([char_embeds]) if char_embeds is not None else char_embeds
            text_lens = np.array([len(texts[0])])
            batchs = [(['tmp'], [raw_texts], speakers, texts, text_lens, max(text_lens))]
            chars_embeds = [char_embeds]

            control_values = pitch_control, energy_control, duration_control
            print('starting speech synthesizing...')
            synthesize(self.model,
                       self.internal_conf,
                       self.vocoder,
                       batchs,
                       chars_embeds,
                       control_values,
                       self.result_path)
            return os.path.join(os.path.abspath(self.result_path), "tmp.wav")

if __name__ == "__main__":
    # 从文件中批量合成语音
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument("--text_file", type=str, required=False, default="",
    #                     help="text input path")
    # parser.add_argument("--output_dir", type=str, required=False, default="",
    #                     help="wav output path")
    # args = parser.parse_args()
    #
    # synthesize_all(args.text_file, args.output_dir)

    # 单句语音合成
    tts = SpeechSynthesis('./config/AISHELL3')
    while True:
        text = input("请输入文本：")
        print(tts.text2speech(text))

    pass
