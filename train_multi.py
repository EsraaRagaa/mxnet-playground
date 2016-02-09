import mxnet as mx
import voc_utils.voc_utils as voc
import logging

dataiter = mx.io.ImageRecordIter(
    path_imgrec='voc_train.rec',
    data_shape=(3, 200, 200),
    path_imglist='voc_train.lst',
    label_width=len(voc.list_image_sets()),
    shuffle=True,
    batch_size=20)

datavaliter = mx.io.ImageRecordIter(
    path_imgrec='voc_val.rec',
    data_shape=(3, 200, 200),
    path_imglist='voc_val.lst',
    label_width=len(voc.list_image_sets()),
    shuffle=True,
    batch_size=20)

# create multilabel net and train from scratch
input_data = mx.symbol.Variable('data')
conv1 = mx.symbol.Convolution(
    data=input_data,
    name='conv1',
    kernel=(20, 20),
    stride=(20, 20),
    pad=(0, 0),
    num_filter=100)
relu1 = mx.symbol.Activation(
    data=conv1, name='relu1', act_type='relu')
top_pool = mx.symbol.Pooling(
    data=relu1, pool_type='max', kernel=(5, 1),
    stride=(2, 2), pad=(0, 0))
out = mx.symbol.SoftmaxOutput(
    data=top_pool, name='softmax', multi_output=True)
net = mx.model.FeedForward(
    symbol=out, num_epoch=10, learning_rate=0.1, momentum=0.9)

logging.basicConfig(level=logging.INFO)

net.fit(X=dataiter, eval_data=datavaliter)
