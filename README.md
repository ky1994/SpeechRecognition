## 1. ASR

基于深度学习的中文语音识别系统


## 2. 声学模型

声学模型采用CTC进行建模，采用CNN-CTC、GRU-CTC、FSMN等模型`model_speech`，采用keras作为编写框架。


## 3. 语言模型

新增基于self-attention结构的语言模型`model_language\transformer.py`，该模型已经被证明有强于其他框架的语言表达能力。

- 论文地址：https://arxiv.org/abs/1706.03762。


基于CBHG结构的语言模型`model_language\cbhg.py`，该模型之前用于谷歌声音合成，移植到该项目中作为基于神经网络的语言模型。

- 原理地址：https://github.com/crownpku/Somiao-Pinyin



## 4. 数据集
包括stc、primewords、Aishell、thchs30四个数据集，共计约430小时, 相关链接：[http://www.openslr.org/resources.php](http://www.openslr.org/resources.php)


|Name | train | dev | test
|- | :-: | -: | -:
|aishell | 120098| 14326 | 7176
|primewords | 40783 | 5046 | 5073
|thchs-30 | 10000 | 893 | 2495
|st-cmd | 10000 | 600 | 2000


若需要使用所有数据集，只需解压到统一路径下，然后设置utils.py中datapath的路径即可。

与数据相关参数在`utils.py`中：
- data_type: train, test, dev
- data_path: 对应解压数据的路径
- thchs30, aishell, prime, stcmd: 是否使用该数据集
- batch_size: batch_size
- data_length: 我自己做实验时写小一些看效果用的，正常使用设为None即可
- shuffle：正常训练设为True，是否打乱训练顺序
```py
def data_hparams():
    params = tf.contrib.training.HParams(
        # vocab
        data_type = 'train',
        data_path = 'data/',
        thchs30 = True,
        aishell = True,
        prime = False,
        stcmd = False,
        batch_size = 1,
        data_length = None,
        shuffle = False)
      return params
```

### 模型识别

使用test.py检查模型识别效果。
模型选择需和训练一致。