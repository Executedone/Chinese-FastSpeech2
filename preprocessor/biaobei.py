import os, re, json, warnings

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from pypinyin import pinyin, Style


def prepare_align(wave_dir, text_file, out_dir):
    sampling_rate = 22050
    max_wav_value = 32768.0
    with open(text_file, 'r', encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f)):
            if i % 2 == 0:
                basename = line.strip().split('\t')[0]
            if i % 2 == 1:
                p = line.strip().split('\t')[0]
                with open(os.path.join(out_dir, basename+'.lab'), 'w', encoding='utf8') as fs:
                    fs.write(p)

                wave_path = os.path.join(wave_dir, basename+'.wav')
                wav, _ = librosa.load(wave_path, sampling_rate)
                wav = wav / max(abs(wav)) * max_wav_value
                wavfile.write(os.path.join(out_dir, basename+'.wav'), sampling_rate, wav.astype(np.int16))

                # print(basename, p)


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


def get_pinyin_prosody(pinyin, text):
    pinyins = pinyin.split(' ')
    texts = re.split(r'#\d', text)
    texts = [re.sub(r'\W+', '', t) for t in texts]
    # assert len(''.join(texts)) == len(pinyins), f"{''.join(texts)}, {pinyins}"
    prosody_pinyin = []
    if len(''.join(texts)) == len(pinyins):
        start = 0
        for t in texts:
            if len(t) >= 1:
                t_pinyin = pinyins[start:start+len(t)]
                prosody_pinyin.append(t_pinyin)
                start += len(t)
    else:
        try:
            assert '儿' in ''.join(texts), f"{''.join(texts)}"
            start = 0
            for t in texts:
                if len(t) >= 1:
                    if t.endswith('儿'):
                        t_pinyin = pinyins[start:start+len(t)-1]
                        start += len(t)-1
                    else:
                        t_pinyin = pinyins[start:start+len(t)]
                        start += len(t)
                    prosody_pinyin.append(t_pinyin)

        except:
            print(''.join(texts))
    return prosody_pinyin


def get_file_based_pinyin_prosody(txt_path, prosody_pinyin_path):
    prosody_pinyins = {}
    with open(txt_path, 'r', encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f)):
            if i % 2 == 0:
                basename, text = line.strip().split('\t')
            if i % 2 == 1:
                p = line.strip().split('\t')[0]
                prosody_pinyin = get_pinyin_prosody(p, text)
                prosody_pinyins[basename] = prosody_pinyin
    with open(prosody_pinyin_path, 'w', encoding='utf8') as f:
        json.dump(prosody_pinyins, f, ensure_ascii=False, indent=2)


def pinyin_prosody2phone_prosody(pinyin_prosody, lexicon):
    phone_prosody = []
    phones = []
    for ps in pinyin_prosody:
        tmp = []
        for p in ps:
            try:
                phones.extend(lexicon[p])
                tmp.extend([0]*len(lexicon[p]))
            except:
                warnings.warn(f'`{p}` not in lexicon, we shall omit it...')
        tmp[0] = 1
        tmp[-1] = 1
        phone_prosody.extend(tmp)
    assert len(phones) == len(phone_prosody), f'phones: {phones}, prosody: {phone_prosody}'
    return phones, phone_prosody


SILENCE = ['sp', 'spn', 'sil']
def align_phone_prosody(std_phones, std_phone_prosody, phones):
    std_index = 0
    phone_prosody = []
    for p in phones:
        if p in SILENCE:
            phone_prosody.append(1)
        else:
            if p == std_phones[std_index]:
                phone_prosody.append(std_phone_prosody[std_index])
                std_index += 1
            else:
                phone_prosody.append(0)
    print('标准：', std_phone_prosody)  # std_phones,
    print('实际：', phone_prosody)  # phones,
    return phone_prosody


def get_file_based_phone_prosody(phones_paths, pinyin_prosody_path, lexicon_path, phone_prosody_path):
    lexicon = read_lexicon(lexicon_path)
    with open(pinyin_prosody_path, 'r') as f:
        pinyins_prosody = json.load(f)

    phones_prosody = {}
    for phones_path in phones_paths:
        with open(phones_path, 'r', encoding='utf8') as f:
            for n, line in enumerate(f):
                line = line.strip().split('|')
                basename, phones = line[0], line[2].replace('{', '').replace('}', '').split(' ')
                print(basename)
                if basename in pinyins_prosody and len(pinyins_prosody[basename]) >= 1:
                    std_phones, std_phone_prosody = pinyin_prosody2phone_prosody(pinyins_prosody[basename], lexicon)
                    phone_prosody = align_phone_prosody(std_phones, std_phone_prosody, phones)
                else:
                    print('未找到basename...')
                    phone_prosody = [1 if p in SILENCE else 0 for p in phones]
                    phone_prosody[0] = 1
                    phone_prosody[-1] = 1
                phones_prosody[basename] = phone_prosody
    print(len(phones_prosody))
    with open(phone_prosody_path, 'w', encoding='utf8') as f:
        json.dump(phones_prosody, f, ensure_ascii=False, indent=2)


def correct_phrase_phone_dict(phrase_phone_path, new_phrase_path):
    with open(phrase_phone_path, 'r', encoding='utf8') as f:
        phrase_phones = json.load(f)
    print(len(phrase_phones))
    new_dict = {}
    count = 0
    for i, (w, p) in enumerate(phrase_phones.items()):
        std_p = [p[0] for p in pinyin(w[-1], style=Style.TONE3, strict=False, neutral_tone_with_five=True)][0]
        ori_p = p[-1]
        try:
            assert len(std_p) == len(ori_p), f'std_p: {std_p}, ori_p: {ori_p}'
            std_ps, ori_ps = re.findall(r'[a-z]', std_p), re.findall(r'[a-z]', ori_p)
            std_pn, ori_pn = re.findall(r'\d', std_p)[0], re.findall(r'\d', ori_p)[0]
            # print(std_ps, std_pn, ori_ps, ori_pn)
            if std_pn == '3' and ori_pn == '2' and std_ps == ori_ps:
                print(i, w, p)
                p[-1] = re.sub(r'2', '3', p[-1])
                print(p)
                count += 1
        except:
            pass
        new_dict[w] = p
    print(len(new_dict), count)
    with open(new_phrase_path, 'w', encoding='utf8') as f:
        json.dump(new_dict, f, ensure_ascii=False, indent=2)


# 汉字扩展，与phone长度一致
def expand_chars(biaobei_text, lex_path, saved_file):
    lexicon = read_lexicon(lex_path)

    expanded_dict = dict()
    with open(biaobei_text, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            if i % 2 == 0:
                sample = line.strip().split('\t')
                basename, text = sample[0], ''.join(sample[1:])
                text = re.sub(r'#\d', '', text)
                new_text = re.sub(r'\W', '', text)
            if i % 2 == 1:
                # print(basename)
                expanded_chars, phones = [], []
                pinyin = line.strip().split('\t')[0].split(' ')
                if len(pinyin) == len(new_text):
                    s = 0
                    for char in text:
                        if re.search(r'\W', char):
                            expanded_chars.append(char)
                            phones.append(['sp'])
                        else:
                            char_p = lexicon[pinyin[s]]
                            char = char * len(char_p)
                            expanded_chars.append(char)
                            phones.append(char_p)
                            s += 1

                elif len(pinyin) < len(new_text) and '儿' in new_text:
                    s = 0
                    skip = False
                    for char in text:
                        if skip:
                            skip = False
                            continue
                        if re.search(r'\W', char):
                            expanded_chars.append(char)
                            phones.append(['sp'])
                        else:
                            # print(char, pinyin, s, lexicon[pinyin[s]])
                            char_p = lexicon[pinyin[s]]
                            if char_p[-1] == 'rr' and re.match(r'e\d', ''.join(char_p[:-1])) is None:
                                char = char * len(char_p[:-1]) + '儿'
                                skip = True
                            else:
                                char = char * len(char_p)
                            expanded_chars.append(char)
                            phones.append(char_p)
                            s += 1
                else:
                    print(basename, text, new_text, pinyin)
                    continue

                inst = {'expanded_chars': expanded_chars, 'phones': phones}
                expanded_dict[basename] = inst
    print(len(expanded_dict))
    with open(saved_file, 'w', encoding='utf8') as fs:
        json.dump(expanded_dict, fs, ensure_ascii=False, indent=2)


# 与训练集中抽取的phone做汉字对齐
def align_to_extract_phones(expanded_chars_phones_file, train_data, dev_data, new_expanded_file):
    std_expanded_dict = {}
    with open(expanded_chars_phones_file, 'r', encoding='utf8') as f:
        data = json.load(f)
    for basename, v in data.items():
        chars = ''.join(v['expanded_chars'])
        ps = v['phones']
        while ps[-1] == ['sp']:
            ps = ps[:-1]
            chars = chars[:-1]
        phones = ' '.join([' '.join(p) for p in ps])
        std_expanded_dict[basename] = {'chars': chars, 'phones': phones}
    print(len(std_expanded_dict))

    with open(train_data, 'r') as f:

        for n, line in enumerate(f):
            line = line.strip().split('|')
            basename, phones = line[0], line[2].replace('{', '').replace('}', '')
            phones = phones.split(' ')
            new_chars = []

            if basename in std_expanded_dict:
                # if len(std_expanded_dict[basename]['chars']) == len(phones.split(' ')):
                #     std_expanded_dict[basename]['same'] = 'true'
                # else:
                #     std_expanded_dict[basename]['same'] = 'false'
                # std_expanded_dict[basename]['train_phones'] = phones

                chars = std_expanded_dict[basename]['chars']
                if len(chars) == len(phones):
                    new_chars = chars
                else:
                    std_phones = std_expanded_dict[basename]['phones']
                    std_phones = std_phones.split(' ')
                    s_std, s_p = 0, 0
                    is_break = False
                    for m in range(max(len(chars), len(phones))):
                        if is_break:
                            is_break = False
                            break
                        # print(chars, std_phones[s_std], phones[s_p])
                        if std_phones[s_std] == phones[s_p]:
                            new_chars.append(chars[s_std])
                            s_std += 1
                            s_p += 1
                        else:
                            if std_phones[s_std] == 'sp' and phones[s_p] != 'sp':
                                s_std += 1
                            elif std_phones[s_std] != 'sp' and phones[s_p] == 'sp':
                                s_p += 1
                                new_chars.append('，')
                                # new_chars.append(chars[s_std])
                            else:
                                print(basename, chars, phones)
                                is_break = True

                if len(new_chars) == len(phones):
                    std_expanded_dict[basename]['same'] = 'true'
                else:
                    std_expanded_dict[basename]['same'] = 'false'
                std_expanded_dict[basename]['new_chars'] = ''.join(new_chars)
                std_expanded_dict[basename]['train_phones'] = ' '.join(phones)

    with open(dev_data, 'r') as f:
        for n, line in enumerate(f):
            line = line.strip().split('|')
            basename, phones = line[0], line[2].replace('{', '').replace('}', '')
            phones = phones.split(' ')
            new_chars = []

            if basename in std_expanded_dict:
                # if len(std_expanded_dict[basename]['chars']) == len(phones.split(' ')):
                #     std_expanded_dict[basename]['same'] = 'true'
                # else:
                #     std_expanded_dict[basename]['same'] = 'false'
                # std_expanded_dict[basename]['train_phones'] = phones

                chars = std_expanded_dict[basename]['chars']
                if len(chars) == len(phones):
                    new_chars = chars
                else:
                    std_phones = std_expanded_dict[basename]['phones']
                    std_phones = std_phones.split(' ')
                    s_std, s_p = 0, 0
                    is_break = False
                    for m in range(max(len(chars), len(phones))):
                        if is_break:
                            is_break = False
                            break
                        # print(chars, std_phones[s_std], phones[s_p])
                        if std_phones[s_std] == phones[s_p]:
                            new_chars.append(chars[s_std])
                            s_std += 1
                            s_p += 1
                        else:
                            if std_phones[s_std] == 'sp' and phones[s_p] != 'sp':
                                s_std += 1
                            elif std_phones[s_std] != 'sp' and phones[s_p] == 'sp':
                                s_p += 1
                                new_chars.append('，')
                                # new_chars.append(chars[s_std])
                            else:
                                print(basename, chars, phones)
                                is_break = True

                if len(new_chars) == len(phones):
                    std_expanded_dict[basename]['same'] = 'true'
                else:
                    std_expanded_dict[basename]['same'] = 'false'
                std_expanded_dict[basename]['new_chars'] = ''.join(new_chars)
                std_expanded_dict[basename]['train_phones'] = ' '.join(phones)

    with open(new_expanded_file, 'w', encoding='utf8') as fs:
        json.dump(std_expanded_dict, fs, ensure_ascii=False, indent=2)


def check_expanded_chars(expanded_file, new_expanded_file):
    with open(expanded_file, 'r', encoding='utf8') as f:
        data = json.load(f)
    for basename, v in data.items():
        if len(v['new_chars']) == len(v['train_phones'].split(' ')):
            v['same'] = 'true'
        else:
            v['same'] = 'false'
        data[basename] = v
    with open(new_expanded_file, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_expanded_length(expanded_file, length_file):
    with open(expanded_file, 'r', encoding='utf8') as f:
        data = json.load(f)
    new_data = dict()
    for basename, v in data.items():
        expanded_chars = v['new_chars']
        text, lengths = [], []
        tmp_c = expanded_chars[0]
        tmp_l = 1
        text.append(tmp_c)
        for i in range(1, len(expanded_chars)):
            if expanded_chars[i] == tmp_c:
                tmp_l += 1
            else:
                lengths.append(tmp_l)
                tmp_c = expanded_chars[i]
                tmp_l = 1
                text.append(tmp_c)
        lengths.append(tmp_l)
        text = ''.join(text)
        # print(text, lengths)
        assert len(text) == len(lengths) and sum(lengths) == len(v['train_phones'].split(' ')), \
            f'basename: {basename}, text: {text}, lengths: {lengths}, train_phones: {v["train_phones"]}, chars: {expanded_chars}'

        new_data[basename] = {'text': text, 'lengths': lengths}

    with open(length_file, 'w', encoding='utf8') as fs:
        json.dump(new_data, fs, ensure_ascii=False, indent=2)


def test_bert_tokenizer(expanded_file):
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('../transformer/prosody_model/')
    with open(expanded_file, 'r', encoding='utf8') as f:
        data = json.load(f)
    for k, v in data.items():
        chars = v['text']
        tokens = tokenizer.tokenize(chars)
        if len(chars) != len(tokens):
            print(k, chars, tokens)


def get_char_embedding_from_bert(expanded_file, model_dir, expand_embeds_file):
    from transformers import BertTokenizer
    from transformer.ProsodyModel import CharEmbedding
    import torch, pickle
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = CharEmbedding(model_dir)
    model.load_state_dict(torch.load('../transformer/prosody_model/best_model.pt', map_location='cpu'))
    model.eval()

    with open(expanded_file, 'r', encoding='utf8') as f:
        data = json.load(f)

    embed_dict = dict()
    for k, v in data.items():
        text, lens = v['text'], v['lengths']
        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        input_masks = [1] * len(input_ids)
        type_ids = [0] * len(input_ids)

        input_ids = torch.LongTensor([input_ids])
        input_masks = torch.LongTensor([input_masks])
        type_ids = torch.LongTensor([type_ids])

        with torch.no_grad():
            embeds = model(input_ids, input_masks, type_ids).squeeze(0)
        assert embeds.size(0) == len(lens)
        expand_vecs = list()
        for vec, length in zip(embeds, lens):
            vec = vec.expand(length, -1)
            expand_vecs.append(vec)
        expand_embeds = torch.cat(expand_vecs, 0)
        print(k, embeds.shape, expand_embeds.shape)
        assert expand_embeds.size(0) == sum(lens)
        embed_dict[k] = expand_embeds

    with open(expand_embeds_file, 'wb') as f:
        pickle.dump(embed_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    wave_dir = '/workspace/raw_data/BB10000/Wave22050/'
    text_file = '/workspace/raw_data/BB10000/ProsodyLabeling/000001-010000.txt'
    out_dir = '/workspace/raw_data/BB10000/corpus/'
    # prepare_align(wave_dir, text_file, out_dir)

    txt_path = '/workspace/raw_data/BB10000/ProsodyLabeling/000001-010000.txt'
    prosody_pinyin_path = '../preprocessed_data/biaobei/prosody_pinyin.json'
    # get_file_based_pinyin_prosody(txt_path, prosody_pinyin_path)

    phones_path = ['../preprocessed_data/biaobei/val.txt',
                   '../preprocessed_data/biaobei/train.txt']
    pinyin_prosody_path = '../preprocessed_data/biaobei/prosody_pinyin.json'
    lexicon_path = '../lexicon/pinyin-lexicon-r.txt'
    phone_prosody_path = '../preprocessed_data/biaobei/prosody_phone.json'
    # get_file_based_phone_prosody(phones_path, pinyin_prosody_path, lexicon_path, phone_prosody_path)

    # pinyin = 'niu1 niu5 you3 xiu1 xi5 ri4 de5'
    # text = '妞妞#2有#1“休息日”的#4。'
    # lexicon = read_lexicon('../lexicon/pinyin-lexicon-r.txt')
    # print(get_pinyin_prosody(pinyin, text))

    phrase_phone_path = '../lexicon/phrase_phones_dict_modified.json'
    new_phrase_path = '../lexicon/phrase_phones_dict.json'
    # correct_phrase_phone_dict(phrase_phone_path, new_phrase_path)

    biaobei_text = '/workspace/raw_data/BB10000/ProsodyLabeling/000001-010000.txt'
    lex_path = '../lexicon/pinyin-lexicon-r.txt'
    saved_file = './std_expanded_chars.json'
    # expand_chars(biaobei_text, lex_path, saved_file)

    expanded_chars_phones_file = './std_expanded_chars.json'
    train_data = '../preprocessed_data/biaobei/train.txt'
    dev_data = '../preprocessed_data/biaobei/val.txt'
    new_expanded_file = './expanded_chars.json'
    # align_to_extract_phones(expanded_chars_phones_file, train_data, dev_data, new_expanded_file)

    expanded_file = './expanded_chars_check.json'
    new_expanded_file = './expanded_chars_check.json'
    # check_expanded_chars(expanded_file, new_expanded_file)

    expanded_file = './expanded_chars_check.json'
    length_file = './expanded_chars_length.json'
    # get_expanded_length(expanded_file, length_file)

    expanded_file = './expanded_chars_length.json'
    # test_bert_tokenizer(expanded_file)

    expanded_file = './expanded_chars_length.json'
    model_dir = '../transformer/prosody_model'
    expand_embeds_file = './expanded_embeds.pkl'
    get_char_embedding_from_bert(expanded_file, model_dir, expand_embeds_file)

    pass