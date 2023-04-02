# Chinese-FastSpeech2
基于[标贝中文标准女声数据](https://www.data-baker.com/data/index/TNtts)继续训练，同时对原论文的[FastSpeech2](https://github.com/ming024/FastSpeech2)模型做了改进，引入了韵律表征以及韵律预测模块，使中文发音更生动且富有节奏 

## 20230402 更新
- 1、加入韵律模型训练代码， 在**BertProsody**目录下
- 2、加入韵律模型训练的数据预处理代码（针对标贝数据，代码未整理，先放出来），在**preprocessor/biaobei.py**里

## 样例
参考samples中生成的音频

## 模型文件
本项目主体架构为FastSpeech2+HifiGAN结构，另外在输入阶段引入了中文文本的韵律向量，因此共有三个模型：fastspeech_model、hifigan_model、prosody_model（[网盘链接](https://pan.baidu.com/s/1aizIt1Hg9Xhb1ttCrbzOvQ)，提取码：qgpi），下载后将模型文件放入指定的目录下：
- 8000.pth.tar  --->  output/ckpt/biaobei/
- generator_universal.pth.tar  --->  hifigan/
- best_model.pt  --->  transformer/prosody_model/

## 预测
提供了两种预测方式：1）python synthesize_all.py；2）http接口调用
- 第一种方式是**交互式**，命令行运行python synthesize_all.py后，输入需要转换的文本，运行后会在代码会在当前工作目录下生成tmp.wav文件；
- 第二种方式是**api调用**，运行tts_server.py，会启动语音转文本的接口，调用该接口可参考TestServer.py，同样生成的音频文件(tmp.wav)会保存在当前工作目录下

## 训练
- 由于本项目参考[FastSpeech2](https://github.com/ming024/FastSpeech2)项目，如果想自定义训练，该项目提供了较为详细的训练方法可供参考；
- 本项目对原方法作了一些优化，优化部分可参考博客: [基于FastSpeech2优化的中文语音合成](https://zhuanlan.zhihu.com/p/585086910)

-----
本项目是出于个人兴趣在语音合成方面做的一些尝试，欢迎大家批评指正，多多交流！
