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
input_data = mx.symbol.Variable(name="data")
# stage 1
conv1 = mx.symbol.Convolution(
    data=input_data, kernel=(11, 11), stride=(4, 4), num_filter=96)
relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
pool1 = mx.symbol.Pooling(
    data=relu1, pool_type="max", kernel=(3, 3), stride=(2, 2))
lrn1 = mx.symbol.LRN(data=pool1, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
# stage 2
conv2 = mx.symbol.Convolution(
    data=lrn1, kernel=(5, 5), pad=(2, 2), num_filter=256)
relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
pool2 = mx.symbol.Pooling(
    data=relu2, kernel=(3, 3), stride=(2, 2), pool_type="max")
lrn2 = mx.symbol.LRN(
    data=pool2, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
# stage 3
conv3 = mx.symbol.Convolution(
    data=lrn2, kernel=(3, 3), pad=(1, 1), num_filter=384)
relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
conv4 = mx.symbol.Convolution(
    data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=384)
relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
conv5 = mx.symbol.Convolution(
    data=relu4, kernel=(3, 3), pad=(1, 1), num_filter=256)
relu5 = mx.symbol.Activation(data=conv5, act_type="relu")
pool3 = mx.symbol.Pooling(
    data=relu5, kernel=(3, 3), stride=(2, 2), pool_type="max")
# stage 4
flatten = mx.symbol.Flatten(data=pool3)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=400)
relu6 = mx.symbol.Activation(data=fc1, act_type="relu")
dropout1 = mx.symbol.Dropout(data=relu6, p=0.5)
shaped = mx.symbol.Reshape(data=fc1, target_shape=(20, 20, 20))
softmax = mx.symbol.SoftmaxOutput(
    data=shaped, name='softmax', multi_output=True)

model_args = {}
lr_factor_epoch = 1
lr_factor = 0.9
num_epochs = 10
# num training images is 5717
epoch_size = 5717 / num_epochs
model_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
            step=max(int(epoch_size * lr_factor_epoch), 1),
            factor=lr_factor)

model = mx.model.FeedForward(
    symbol=softmax, num_epoch=num_epochs,
    learning_rate=0.1, momentum=0.9,
    wd=0.00001,
    ctx=[mx.gpu()])

logging.basicConfig(level=logging.INFO)

model.fit(X=dataiter, eval_data=datavaliter)
