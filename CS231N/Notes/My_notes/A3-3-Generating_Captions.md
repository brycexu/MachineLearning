A3-3-Generating_Captions
图片注释生成


词嵌入
Word embedding
我们将每个词用一个向量表示
词表中的每一个词都将和一个向量关联，这些向量则会和系统的其余部分一样进行训练

前向传播
>
输入：
x - (N,T)-->(num_train,length) - N张图片，每张图片用T长度的句子描述
w - (V,D)-->(num_words,vector) - 词典，V个单词，每一行对应着此单词的描述向量
输出：
out - (N,T,D)
cache
>>>
def word_embedding_forward(x, W):
    out = W[x, :]
    cache = x, W
    return out, cache
>>>
反向传播
>
输入：
dout - (N,T,D)
cache
输出：
dW - (V,D)
>>>
def word_embedding_backward(dout, cache):
    x, W = cache
    dW=np.zeros_like(W)
    np.add.at(dW, x, dout)
    return dW
>>>


