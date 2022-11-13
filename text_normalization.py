''' 
_*_ coding: utf-8 _*_
Date: 2021/11/22
Author: 
Intent:
'''

import re


POLY_PHRASE = {
        '我行': [['wǒ'], ['háng']],
        '开户行': [['kāi'], ['hù'], ['háng']],
        '还款': [['huán'], ['kuǎn']],
        '行货': [['háng'], ['huò']],
        '发卡行': [['fā'], ['kǎ'], ['háng']],
        '茧行': [['jiǎn'], ['háng']],
        '还呗': [['huán'], ['bei']],
        '人行': [['rén'], ['háng']],
        '各行': [['gè'], ['háng']],
        '行号': [['háng'], ['hào']],
        '素朴': [['sù'], ['pǔ']]
}


NUM_DICT = {
        '0': '零',
        '1': '一',
        '2': '二',
        '3': '三',
        '4': '四',
        '5': '五',
        '6': '六',
        '7': '七',
        '8': '八',
        '9': '九',
        '10': '十',
        '100': '百',
        '1000': '千',
        '-': '，'
}


TIME = {
        ':': '点',
        '30': '半',
        '-': '至',
        '~': '至'
}


SPECIAL_SYMBOLS = {
        '.': '点',
        '%': '百分之'
}


PUNCTRUATION = [',', ';', '"', '!', '！', '，', '。', '；', '：', '’', '“', '？',
                  '、', '?', '（', '）', '《', '》', '(', ')', '、', '<', '>', '_',
                  '——', '+', '[', ']']


SINGLE_NUM_PATTERN = re.compile(r'(\d{3,20}-*)+\d{2,10}')

PHRASE_NUM_PATTERN = re.compile(r'(\d{1,2}-\d{1,2})|(\d{1,4}\.\d{1,4})|(\d{1,4})|(—55)')

TIME_PATTERN = re.compile(r'(\d{1,2}:\d{2}[-~]\d{1,2}:\d{2})|(\d{1,2}:30)|(\d{1,2}:00)|(\d{1,2}:\d{2})')

PERCENTAGE_PATTERN = re.compile(r'\d+(\.\d+)*%')


def get_num_phones(num):
    phones = []
    num_len = len(num)
    if num_len == 2 and num[0] == '1':
        phones.append(NUM_DICT['10'])
        if num[1] != '0':
            phones.append(NUM_DICT[num[1]])
    else:
        for i, n in enumerate(num):
            if num[i:] == '0' * (num_len - i) or num[:(i + 1)] == '0' * (i + 1):
                continue
            else:
                if len(phones) > 0 and phones[-1] == '零' and n == '0':
                    continue
                else:
                    phones.append(NUM_DICT[n])
            if n == '0':
                continue
            if num_len - i == 4:
                phones.append(NUM_DICT['1000'])
            elif num_len - i == 3:
                phones.append(NUM_DICT['100'])
            elif num_len - i == 2:
                phones.append(NUM_DICT['10'])
            else:
                pass
    return phones


def get_time_phones(time):
    phones = []
    assert ':' in time
    h, m = time.split(':')
    h_phone = get_num_phones(h)
    if m == '30':
        m_phone = [TIME['30']]
    elif m == '00':
        m_phone = None
    else:
        m_phone = get_num_phones(m)
    phones.extend(h_phone)
    phones.append(TIME[':'])
    if m_phone is not None:
        phones.extend(m_phone)
    return phones


def get_percent_phones(percentage):
    phones = SPECIAL_SYMBOLS['%'].split(' ')
    s = percentage.replace('%', '')

    if '.' in s:
        pre, post = s.split('.')
        comma = SPECIAL_SYMBOLS['.']
    else:
        pre, post = s, ''
        comma = None

    if pre == '0':
        phones.append(NUM_DICT['0'])
    else:
        phones.extend(get_num_phones(pre))

    if comma is not None:
        phones.append(comma)
    phones.extend([NUM_DICT[n] for n in post if n in NUM_DICT])
    return phones


def get_special_phones(phrase):
    phones = []

    if phrase == '2':
        return ['两']

    if phrase == '—55':
        return ['至', '五', '十', '五']

    if PERCENTAGE_PATTERN.match(phrase):
        return get_percent_phones(phrase)

    if TIME_PATTERN.match(phrase):
        if '-' in phrase or '~' in phrase:
            b = '-' if '-' in phrase else '~'
            temp = phrase.split(b)
            pre_t, post_t = temp[0], temp[1]
            pre_t_phones = get_time_phones(pre_t)
            post_t_phones = get_time_phones(post_t)
            phones.extend(pre_t_phones)
            phones.append(TIME[b])
            phones.extend(post_t_phones)
        else:
            t_phones = get_time_phones(phrase)
            phones.extend(t_phones)
        return phones

    if SINGLE_NUM_PATTERN.match(phrase):
        phones.extend([NUM_DICT[num] for num in phrase if num in NUM_DICT])
        return phones

    if PHRASE_NUM_PATTERN.match(phrase):
        if '-' in phrase:
            nums = phrase.split('-')
            for i, num in enumerate(nums):
                if len(num) > 0:
                    phones.extend(get_num_phones(num))
                    if i < len(nums) - 1:
                        phones.append(TIME['-'])
        elif '.' in phrase:
            temp = phrase.split('.')
            phrase_num = temp[0]
            single_num = temp[1]
            phones.extend(get_num_phones(phrase_num))
            phones.append(SPECIAL_SYMBOLS['.'])
            phones.extend([NUM_DICT[n] for n in single_num if n in NUM_DICT])
        else:
            num = re.match(r'\d+', phrase).group()
            phones.extend(get_num_phones(num))
        return phones

    return phones


def pattern_normalize(text, pattern):
    normalized_text = []
    s, e = 0, 0
    for x in pattern.finditer(text):
        phrase, start, end = x.group(), x.start(), x.end()
        e = start
        normalized_text.append(text[s:e])
        s = end
        normalized_text.append(''.join(get_special_phones(phrase)))
    normalized_text.append(text[s:len(text)])

    text = ''.join(normalized_text)
    return text


def eliminate_duplicate_punctuation(text):
    text = re.sub(r'\W+', '，', text)
    return text


def text_normalization(text):
    text = pattern_normalize(text, PERCENTAGE_PATTERN)
    text = pattern_normalize(text, TIME_PATTERN)
    text = pattern_normalize(text, SINGLE_NUM_PATTERN)
    text = pattern_normalize(text, PHRASE_NUM_PATTERN)
    text = eliminate_duplicate_punctuation(text)
    return text


if __name__ == "__main__":

    pass