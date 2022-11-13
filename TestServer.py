''' 
_*_ coding: utf-8 _*_
Date: 2022/3/4
Author: 
Intent:
'''

import requests

def test_api(text):
    url = "http://127.0.0.1:5876/TextToSpeech"
    data = {'text': text}
    res = requests.post(url, data).json()
    print(res)
    return res

if __name__ == "__main__":
    text = '踢足球的人'
    test_api(text)

    pass
