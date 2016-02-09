import mxnet as mx
import voc_utils.voc_utils as voc
import logging

dataiter = mx.io.ImageRecordIter(
    path_imgrec='voc_train.rec',
    data_shape=(3, 224, 224),
    path_imglist='voc_train.lst',
    label_width=len(voc.list_image_sets()),
    shuffle=True,
    batch_size=20)

datavaliter = mx.io.ImageRecordIter(
    path_imgrec='voc_val.rec',
    data_shape=(3, 224, 224),
    path_imglist='voc_val.lst',
    label_width=len(voc.list_image_sets()),
    shuffle=True,
    batch_size=20)


def ConvFactory(data, num_filter, kernel, stride=(1, 1), pad=(0, 0),
                name=None, suffix=''):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter,
                                 kernel=kernel, stride=stride, pad=pad,
                                 name='conv_%s%s' % (name, suffix))
    act = mx.symbol.Activation(data=conv, act_type='relu',
                               name='relu_%s%s' % (name, suffix))
    return act


def InceptionFactory(data, num_1x1, num_3x3red, num_3x3, num_d5x5red,
                     num_d5x5, pool, proj, name):
    # 1x1
    c1x1 = ConvFactory(data=data, num_filter=num_1x1, kernel=(1, 1),
                       name=('%s_1x1' % name))
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red,
                        kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3),
                       pad=(1, 1), name=('%s_3x3' % name))
    # double 3x3 reduce + double 3x3
    cd5x5r = ConvFactory(data=data, num_filter=num_d5x5red, kernel=(1, 1),
                         name=('%s_5x5' % name), suffix='_reduce')
    cd5x5 = ConvFactory(data=cd5x5r, num_filter=num_d5x5, kernel=(5, 5),
                        pad=(2, 2), name=('%s_5x5' % name))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(1, 1),
                                pad=(1, 1), pool_type=pool,
                                name=('%s_pool_%s_pool' % (pool, name)))
    cproj = ConvFactory(data=pooling, num_filter=proj, kernel=(1, 1),
                        name=('%s_proj' % name))
    # concat
    concat = mx.symbol.Concat(*[c1x1, c3x3, cd5x5, cproj],
                              name='ch_concat_%s_chconcat' % name)
    return concat

input_data = mx.sym.Variable("data")
conv1 = ConvFactory(input_data, 64, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                    name="conv1")
pool1 = mx.sym.Pooling(conv1, kernel=(3, 3), stride=(2, 2), pool_type="max")
conv2 = ConvFactory(pool1, 64, kernel=(1, 1), stride=(1, 1), name="conv2")
conv3 = ConvFactory(conv2, 192, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                    name="conv3")
pool3 = mx.sym.Pooling(conv3, kernel=(3, 3), stride=(2, 2), pool_type="max")

in3a = InceptionFactory(pool3, 64, 96, 128, 16, 32, "max", 32, name="in3a")
in3b = InceptionFactory(in3a, 128, 128, 192, 32, 96, "max", 64, name="in3b")
pool4 = mx.sym.Pooling(in3b, kernel=(3, 3), stride=(2, 2), pool_type="max")
in4a = InceptionFactory(pool4, 192, 96, 208, 16, 48, "max", 64, name="in4a")
in4b = InceptionFactory(in4a, 160, 112, 224, 24, 64, "max", 64, name="in4b")
in4c = InceptionFactory(in4b, 128, 128, 256, 24, 64, "max", 64, name="in4c")
in4d = InceptionFactory(in4c, 112, 144, 288, 32, 64, "max", 64, name="in4d")
in4e = InceptionFactory(in4d, 256, 160, 320, 32, 128, "max", 128, name="in4e")
pool5 = mx.sym.Pooling(in4e, kernel=(3, 3), stride=(2, 2), pool_type="max")
in5a = InceptionFactory(pool5, 256, 160, 320, 32, 128, "max", 128, name="in5a")
in5b = InceptionFactory(in5a, 384, 192, 384, 48, 128, "max", 128, name="in5b")
pool6 = mx.sym.Pooling(in5b, kernel=(7, 7), stride=(1, 1), pool_type="avg")
flatten = mx.sym.Flatten(data=pool6)
fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=400)
shaped = mx.symbol.Reshape(data=fc1, target_shape=(20, 20, 20))
# relu6 = mx.symbol.Activation(data=shaped, act_type="relu")
# dropout1 = mx.symbol.Dropout(data=relu6, p=0.5)
softmax = mx.symbol.SoftmaxOutput(data=shaped, name='softmax',
                                  multi_output=True)

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
