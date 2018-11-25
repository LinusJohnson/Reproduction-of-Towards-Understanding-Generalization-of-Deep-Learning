import tensorflow as tf
import logging
# import copy
from functools import reduce
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import constants
import gc
import os
import gzip
import shutil
import tempfile
from six.moves import urllib
from memory_profiler import profile
import psutil
import re
import subprocess


# ------- DATBASE FUNCTIONS ------- #
# from https://github.com/tensorflow/models/blob/master/official/mnist/dataset.py


def read32(bytestream):
    """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def check_image_file_header(filename):
    """Validate that filename corresponds to images for the MNIST dataset."""
    with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        read32(f)  # num_images, unused
        rows = read32(f)
        cols = read32(f)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST file %s' %
                             (magic, f.name))
        if rows != 28 or cols != 28:
            raise ValueError(
                'Invalid MNIST file %s: Expected 28x28 images, found %dx%d' %
                (f.name, rows, cols))


def check_labels_file_header(filename):
    """Validate that filename corresponds to labels for the MNIST dataset."""
    with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        read32(f)  # num_items, unused
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST file %s' %
                             (magic, f.name))


def download(directory, filename):
    """Download (and unzip) a file from the MNIST dataset if not already done."""
    filepath = os.path.join(directory, filename)
    if tf.gfile.Exists(filepath):
        return filepath
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + filename + '.gz'
    _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
    print('Downloading %s to %s' % (url, zipped_filepath))
    urllib.request.urlretrieve(url, zipped_filepath)
    with gzip.open(zipped_filepath, 'rb') as f_in, \
            tf.gfile.Open(filepath, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(zipped_filepath)
    return filepath


# ------- SUPPORT FUNCTIONS ------- #

def optimize_frozen_graph(preserved_nodes):
    subprocess.call(
        ['python', '-m', 'tensorflow.python.tools.optimize_for_inference', '--input '+constants.FROZEN_GRAPH, ' --output '+constants.OPT_GRAPH, ' --input_names=' + preserved_nodes[-1], ' --output_names='+preserved_nodes[0][:-2]+','+preserved_nodes[1][:-2]])


def clear_session(sess):
    sess.close()
    gc.collect()


def initialize_model_res_dict():
    model_res = dict()
    model_res['min_log_eig'] = np.inf
    model_res['max_log_eig'] = -np.inf
    model_res['eigenvalues'] = []
    model_res['log_top_fifty_eigs'] = []
    model_res['top_log_eig_sum'] = []
    model_res['test_acc'] = []
    model_res['noise_mix'] = []
    model_res['train_acc'] = []
    model_res['exec_time'] = []
    model_res['M'] = []
    return model_res


def get_params(prob,
               def_params=constants.DEFAULT_PARAMS,
               params_to_ev=constants.PARAMS_TO_EVALUATE):
    params = dict(def_params)
    params.update(params_to_ev[prob])
    return params


def get_session_name(prob_key, sample):
    return prob_key + '_' + str(sample)


def create_plots(fig_dir=constants.FIG_DIR, restore=False):
    plot = dict()
    plot['fig_dir'] = fig_dir
    tf.gfile.MakeDirs(fig_dir)
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    plot['axes'] = [ax1, ax2, ax3]
    plot['figs'] = [fig1, fig2, fig3]
    plot['axes'][0].set_xlabel(r'$\sum_{i=1}^{50}\log_{10}(\lambda_i)$')
    plot['axes'][0].set_ylabel('Test Accuracy, in %')
    plot['axes'][1].set_xlabel(r'$n$')
    plot['axes'][1].set_ylabel(r'$\log_{10}(\lambda_n)$')
    plot['axes'][2].set_xlabel(r'$\lambda$')
    plot['handles'] = [[], [], []]
    plot['names'] = ['basin', 'top_eigs', 'histogram']

    if restore:
        for name, ax in zip(plot['names'], plot['axes']):
            path_to_prev_plots = os.path.join(
                constants.PREV_PLOTS_DIR, name+'.png')
            if tf.gfile.Exists(path_to_prev_plots):
                prev_plot = plt.imread(path_to_prev_plots)
                ax.imshow(prev_plot)
    return plot


def update_plots(res, params, sess_name, plot, c_scale=1):
    figs = plot['figs']
    axes = plot['axes']
    handles = plot['handles']
    fig_dir = plot['fig_dir']
    names = plot['names']
    tf.logging.info('Plotting')
    cmap = plt.get_cmap('Dark2')
    gc.collect()
    handles[0].append(axes[0].scatter(
        res['top_log_eig_sum'][-1],
        res['test_acc'][-1],
        c=cmap(res['noise_mix'][-1]*c_scale),
        s=20,
        label='{:.2%}'.format(res['noise_mix'][-1])))

    handles[1].append(axes[1].scatter(
        np.arange(len(res['log_top_fifty_eigs'][-1])),
        res['log_top_fifty_eigs'][-1][::-1],
        c=cmap(res['noise_mix'][-1]*c_scale),
        s=20,
        label='{:.2%}'.format(res['noise_mix'][-1])))

    # no handle for hist
    axes[2].hist(
        res['eigenvalues'][-1],
        bins=300,
        color=cmap(res['noise_mix'][-1]*c_scale),
        alpha=1,
        histtype=u'step',
        log=True,
        label='{:.2%}'.format(res['noise_mix'][-1]))

    for i in range(len(handles)):
        handle, ax = handles[i], axes[i]
        if ax.legend_ is not None:
            ax.legend_.remove()
        if i != 2:
            ax.legend(handles=handle)
        else:
            ax.legend()

    tf.logging.info('Saving plots')
    formats_to_save = ['pdf', 'svg', 'eps', 'png']
    dpi = 1200
    gc.collect()
    for fmt in formats_to_save:
        tf.gfile.MakeDirs(fig_dir + '/' + fmt)
        tf.logging.info('Saving figures in %s format', fmt)
        for fig, name in zip(figs, names):
            tf.logging.info('Saving %ss', name)
            gc.collect()
            fig.savefig(
                fig_dir + '/' + fmt + '/' + name + '.' + fmt,
                bbox_inches='tight',
                format=fmt,
                dpi=dpi)
    tf.logging.info('Plots completed')
    gc.collect()


def setup_logging(log_dir=constants.LOG_DIR):
    tf.logging.set_verbosity(tf.logging.ERROR)
    log = logging.getLogger('tensorflow')

    logstd = log.handlers[0]
    log.removeHandler(logstd)

    tf.gfile.MakeDirs(log_dir)
    tf.reset_default_graph()

    fh = logging.FileHandler('tensorflow.log')
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    log.addHandler(fh)
    log.propagate = False
    fh.propagate = False

    tf.logging.set_verbosity(tf.logging.DEBUG)

    tf.logging.info('\n\nStarting run\n')


def calc_stats(res, prob, log=True):
    res[prob]['steps']['mean'] = np.mean(res[prob]['steps']['samples'])
    res[prob]['steps']['std'] = np.std(res[prob]['steps']['samples'], ddof=1)
    res[prob]['test_accuracy']['mean'] = np.mean(
        res[prob]['test_accuracy']['samples'])
    res[prob]['test_accuracy']['std'] = np.std(
        res[prob]['test_accuracy']['samples'], ddof=1)
    res[prob]['train_accuracy']['mean'] = np.mean(
        res[prob]['train_accuracy']['samples'])
    res[prob]['train_accuracy']['std'] = np.std(
        res[prob]['train_accuracy']['samples'], ddof=1)
    res[prob]['exec_time']['mean'] = np.mean(res[prob]['exec_time']['samples'])
    res[prob]['exec_time']['std'] = np.std(
        res[prob]['exec_time']['samples'], ddof=1)
    if log:
        tf.logging.info('For %s samples:', constants.NUM_SAMPLES)
        tf.logging.info('\tSteps Mean: %s', res[prob]['steps']['mean'])
        tf.logging.info('\tSteps Std:  %s', res[prob]['steps']['std'])
        tf.logging.info('\tTrain Error Mean: %s',
                        res[prob]['train_accuracy']['mean'])
        tf.logging.info('\tTrain Error Std: %s',
                        res[prob]['train_accuracy']['std'])
        tf.logging.info('\tTest Accuracy Mean: %s',
                        res[prob]['test_accuracy']['mean'])
        tf.logging.info('\tTest Accuracy Std:  %s',
                        res[prob]['test_accuracy']['std'])
        tf.logging.info('\tExecution Time Mean: %s',
                        res[prob]['exec_time']['mean'])
        tf.logging.info('\tExecution Time Std:  %s',
                        res[prob]['exec_time']['std'])


def init_res_dict(params_to_ev=constants.PARAMS_TO_EVALUATE):
    results = dict()
    for prob in params_to_ev.keys():
        results[prob] = {
            'test_accuracy': dict(),
            'train_accuracy': dict(),
            'steps': dict(),
            'exec_time': dict()
        }
        results[prob]['test_accuracy']['samples'] = np.zeros(
            constants.NUM_SAMPLES)
        results[prob]['train_accuracy']['samples'] = np.zeros(
            constants.NUM_SAMPLES)
        results[prob]['steps']['samples'] = np.zeros(constants.NUM_SAMPLES)
        results[prob]['exec_time']['samples'] = np.zeros(constants.NUM_SAMPLES)
    return results


def add_sample_to_res(model_res, prob, sample, results):
    results[prob]['steps']['samples'][sample] = model_res['M'][-1]
    results[prob]['exec_time']['samples'][sample] = model_res['exec_time'][-1]
    results[prob]['train_accuracy']['samples'][sample] = model_res[
        'train_acc'][-1]
    results[prob]['test_accuracy']['samples'][sample] = model_res['test_acc'][
        -1]


# ------- DATA FUNCTIONS ------- #

def get_weakened_data(params, fake=False):
    d = {'test': dict(), 'train': dict(), 'hessian': dict()}

    with tf.name_scope('Training_dataset'):
        d['train']['dataset'], train_dataset_size = dataset(
            constants.DATA_DIR,
            'train-images-idx3-ubyte',
            'train-labels-idx1-ubyte',
            params,
            train=True)
        with tf.name_scope('Hessian_dataset'):
            d['hessian']['dataset'] = d['train']['dataset'].cache().shuffle(
                buffer_size=50000).batch(constants.HESSIAN_BATCH_SIZE)
        d['train']['dataset'] = d['train']['dataset'].cache().shuffle(
            buffer_size=50000).batch(constants.BATCH_SIZE)

    with tf.name_scope('Testing_dataset'):
        d['test']['dataset'], test_dataset_size = dataset(
            constants.DATA_DIR,
            't10k-images-idx3-ubyte',
            't10k-labels-idx1-ubyte',
            params,
            train=False)
        d['test']['dataset'] = d['test']['dataset'].cache().shuffle(
            buffer_size=50000).batch(constants.BATCH_SIZE)

    d['train']['num_batches'] = train_dataset_size / constants.BATCH_SIZE
    d['test']['num_batches'] = test_dataset_size / constants.BATCH_SIZE
    d['hessian']['num_batches'] = d['train']['num_batches']
    if fake:
        d['train']['dataset'] = d['train']['dataset'].take(1)
        d['test']['dataset'] = d['test']['dataset'].take(1)
        d['hessian']['dataset'] = d['hessian']['dataset'].take(1)
        d['train']['num_batches'] = 1
        d['test']['num_batches'] = 1
        d['hessian']['num_batches']

    return d


def dataset(directory, images_file, labels_file, params, train):
    """Download, parse, and preprocess MNIST dataset."""

    images_file = download(directory, images_file)
    labels_file = download(directory, labels_file)

    check_image_file_header(images_file)
    check_labels_file_header(labels_file)

    def decode_image(image):
        # Normalize from [0, 255] to [0.0, 1.0]
        image = tf.decode_raw(image, tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [28, 28, 1])
        return image / 255.0

    def decode_label(label):
        label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
        label = tf.reshape(label, [])  # label is a scalar
        return tf.one_hot(tf.to_int32(label), 10)

    with tf.name_scope('Underlying_Dataset'):
        images = tf.data.FixedLengthRecordDataset(
            images_file, 28 * 28, header_bytes=16).map(decode_image)
        labels = tf.data.FixedLengthRecordDataset(
            labels_file, 1, header_bytes=8).map(decode_label)

    """ Calculate the size of the data by looking at the number of labels """
    totalBytes = os.path.getsize(labels_file)
    bytesPerRecord = 2  # 8 bit
    num_labels = int(totalBytes / bytesPerRecord)
    dataset_size = num_labels
    tf.logging.info('dataset size: %s', dataset_size)

    """ Shuffle a subset of the labels """
    if train:
        with tf.name_scope('Weakening_Data'):
            part_real_labels = 1 - params['noise_mix']

            num_real_labels = int(part_real_labels * num_labels)
            num_attack_labels = int((1 - part_real_labels) * num_labels)
            num_attack_labels += num_labels - \
                (num_real_labels + num_attack_labels)

            real_labels = labels.take(num_real_labels)
            attack_labels = labels.skip(num_real_labels).take(
                num_attack_labels).shuffle(buffer_size=50000)
            attack_images = images.skip(
                num_real_labels).take(num_attack_labels)
            real_images = images.take(num_real_labels)
            images = real_images.concatenate(attack_images)
            labels = real_labels.concatenate(attack_labels)

    with tf.name_scope('Final_Zipped_Dataset'):
        final_dataset = tf.data.Dataset.zip((images, labels))

    return final_dataset, dataset_size


# ------- MODEL FUNCTIONS ------- #


def get_eigs(mat):
    with tf.name_scope('eigenvalues'):
        eigs = tf.self_adjoint_eigvals(mat)
        eigs = tf.reshape(eigs, [-1])
    return eigs


def hessian(optimizer, cost):
    tf.logging.info('Building Hessian')
    with tf.name_scope('hessian'):
        parameters = tf.trainable_variables()
        tf.logging.info('Total number of parameters: %s',
                        tf.concat([tf.reshape(param, [-1]) for param in parameters], 0).shape)
        # tf.logging.info('Parameters: %s', parameters)
        with tf.name_scope('grad_calc'):
            gradient_param_pair = optimizer.compute_gradients(cost)
            # tf.logging.info('Gradient & Parameters: %s', gradient_param_pair)
            gradient = tf.concat(
                [
                    tf.reshape(grad, [-1])
                    for grad, param in gradient_param_pair
                ],
                axis=0)
        with tf.name_scope('outer_param_loop'):
            # hessian_list = []
            hessian = None
            for param in parameters:
                with tf.name_scope('inner_grad_loop_' + param.op.name):
                    with tf.name_scope('while_loop'):
                        _, hessian_ = tf.while_loop(
                            lambda j, _: j < tf.size(gradient),
                            lambda j, result: (
                                j + 1,
                                result.write(j,
                                             tf.gradients(gradient[j],
                                                          param)[0]
                                             )),
                            [
                                tf.constant(0, constants.INT_DTYPE),
                                tf.TensorArray(param.dtype, tf.size(gradient))
                            ]
                        )
                    with tf.name_scope('reshaping'):
                        hessian_ = hessian_.stack()
                        hessian_ = tf.reshape(
                            hessian_, [tf.size(gradient),
                                       tf.size(param)])
                        if hessian is None:
                            hessian = hessian_
                        else:
                            hessian = tf.concat([hessian, hessian_], 1)
    return hessian


def neural_network_model(model, params):
    tf.logging.info('Compiling Network')
    with tf.name_scope('model'):
        initializer = tf.initializers.random_normal
        activation_fcn = params['ActFun']
        activation_fcn_output = None
        num_output_nodes = 10
        layer_inp = model['streams']['images']

        with tf.name_scope('hidden_layers'):
            for i, layer in enumerate(range(params['Layers'])):
                with tf.name_scope('hidden_layer_' + str(i)):
                    if params['LayerTypes'][layer] is tf.layers.dense:
                        if tf.rank(layer_inp) != 1:
                            layer_inp = tf.contrib.layers.flatten(layer_inp)
                        hidden_layer = params['LayerTypes'][layer](
                            inputs=layer_inp,
                            units=params['K'][layer],
                            activation=activation_fcn,
                            kernel_initializer=initializer())
                    if params['LayerTypes'][layer] is tf.layers.conv2d:
                        hidden_layer = params['LayerTypes'][layer](
                            inputs=layer_inp,
                            filters=params['K'][layer],
                            kernel_size=[5, 5],
                            activation=activation_fcn,
                            kernel_initializer=initializer())
                    if params['LayerTypes'][layer] is tf.layers.max_pooling2d:
                        hidden_layer = params['LayerTypes'][layer](
                            inputs=layer_inp,
                            pool_size=[params['K'][layer], params['K'][layer]],
                            strides=params['K'][layer])
                layer_inp = hidden_layer

        with tf.name_scope('output'):
            logits = tf.layers.dense(
                inputs=hidden_layer,
                units=num_output_nodes,
                activation=activation_fcn_output,
                use_bias=False,
                kernel_initializer=initializer())

        with tf.name_scope('cost'):
            cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logits, labels=model['streams']['labels']))
        model['placeholders']['cost_value'] = tf.placeholder(
            tf.float32, shape=())
        cost_sum = tf.summary.scalar(
            'cost', model['placeholders']['cost_value'])

        if (params['Optimizer'] == tf.train.GradientDescentOptimizer):
            optimizer = params['Optimizer'](constants.dt)
        else:
            optimizer = params['Optimizer']()

        with tf.name_scope('train'):
            train = optimizer.minimize(
                cost, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

        with tf.name_scope('accuracy'):
            with tf.name_scope('prediction_is_correct'):
                prediction_is_correct = tf.equal(
                    tf.argmax(logits, 1), tf.argmax(model['streams']['labels'], 1))
            accuracy = tf.reduce_mean(
                tf.cast(prediction_is_correct, constants.DTYPE))
        model['placeholders']['accuracy_value'] = tf.placeholder(
            tf.float32, shape=())
        accuracy_sum = tf.summary.scalar(
            'accuracy', model['placeholders']['accuracy_value'])
        hes = hessian(optimizer, cost)
        tf.logging.info('Completed network compilation')

        with tf.name_scope('hessian_eigenvalue_summary'):
            model['placeholders']['hessian'] = tf.placeholder(constants.DTYPE)
            hess_eigs = get_eigs(model['placeholders']['hessian'])
            hessian_eig_hist_summary = tf.summary.histogram(
                'hessian_eigenvalues', hess_eigs)

        model['ops'] = {
            'train': train,
            'cost': cost,
            'accuracy': accuracy,
            'hessian': hes,
            'hess_eigs': hess_eigs,
            'hessian_eig_hist_summary': hessian_eig_hist_summary,
            'cost_summary': cost_sum,
            'accuracy_summary': accuracy_sum
        }


def sum_over_dataset(name, model, op, sess, max_num_batches=-1, writer=None, collect_metadata=False, verbose=True):
    process = psutil.Process(os.getpid())
    gc.collect()
    if max_num_batches == -1 or constants.ALLOW_FULL_SAMPLING:
        max_num_batches = model['data'][name]['num_batches']
    sess.run(model['data'][name]['iterator'].initializer)
    with tf.name_scope('sum_over_dataset'):
        op_shape = tuple(dim.value for dim in op.shape)
        op_size = reduce(lambda x, y: x * y, op_shape, 1)
        if verbose:
            tf.logging.info(
                'Summing over dataset %s with %s/%s batches', name, max_num_batches, model['data'][name]['num_batches'])
            tf.logging.info(
                'Operation name: %s, shape: %s, size: %s, dtype: %s, size: %s, memory: %s B, %s GB',
                op.name, op_shape, op_size, op.dtype, op.dtype.size,
                op_size * op.dtype.size, op_size * op.dtype.size / 1e9)
        batch_val = np.zeros(op_shape)

        n = 0
        run_metadata = None
        run_options = None
        if collect_metadata and (writer is not None):
            run_metadata = tf.RunMetadata()
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        while True:
            try:
                n += 1
                # avoid spamming the log
                if not constants.ALLOW_FULL_SAMPLING and not (max_num_batches == model['data'][name]['num_batches']) and verbose:
                    tf.logging.info('Summing over batch num: %s', n)
                    tf.logging.info('Memory currently used: %s GB',
                                    process.memory_info().rss / 1e9)
                if op_size > 1:
                    batch_val += sess.run(
                        op,
                        feed_dict={
                            model['placeholders']['iterator_handle']:
                            model['data'][name]['handle']
                        },
                        options=run_options,
                        run_metadata=run_metadata)
                else:
                    batch_val += sess.run(
                        op,
                        feed_dict={
                            model['placeholders']['iterator_handle']:
                            model['data'][name]['handle']
                        })
                if collect_metadata and (writer is not None):
                    tf.logging.info('Writing metadata')
                    writer.add_run_metadata(run_metadata,
                                            '%s_%s' % (op.name, time.time()))
                    writer.flush()
                # allow resticting the sampling if the computation is to big
                if (n >= max_num_batches) and not constants.ALLOW_FULL_SAMPLING:
                    break
            except tf.errors.OutOfRangeError:
                break

    tf.logging.info('Completed summing over dataset')
    tf.logging.info('Memory currently used: %s GB',
                    process.memory_info().rss / 1e9)
    del op_shape, op_size
    gc.collect()
    tf.logging.info('After gc.collect(): %s GB',
                    process.memory_info().rss / 1e9)
    return batch_val


# ------- TRAINING FUNCTIONS ------- #


def early_stopping(train_costs, test_cost, test_cost_min):
    tf.logging.info('latest train costs: %s', train_costs)
    if np.isinf(test_cost) or np.isinf(test_cost_min) or np.isinf(train_costs).any():
        return False
    else:
        P = 1000 * (np.sum(train_costs) /
                    (len(train_costs) * np.min(train_costs)) - 1)
        GL = 100 * ((test_cost / test_cost_min) - 1)
        tf.logging.info('Test Cost: %s, Min: %s, GL: %s, P: %s, GL/P: %s',
                        test_cost, test_cost_min, GL, P, GL / P)
        return GL / P > 3


def update_cost_record(new_cost, costs):
    for i in range(len(costs)):
        if i == len(costs) - 1:
            costs[i] = new_cost
        else:
            costs[i] = costs[i+1]


def eval_scores(m, model, writers, name, sess, verbose=True):
    tf.logging.info('Evaluating ' + name + ' scores')

    # approximate average values over whole dataset
    cost = sum_over_dataset(
        name, model, model['ops']['cost'], sess, verbose=verbose) / model['data'][name]['num_batches']
    acc = sum_over_dataset(
        name, model, model['ops']['accuracy'], sess, verbose=verbose) / model['data'][name]['num_batches']

    writers[name].add_summary(sess.run(model['ops']['accuracy_summary'], feed_dict={
        model['placeholders']['accuracy_value']: acc}), m)
    writers[name].add_summary(sess.run(model['ops']['cost_summary'], feed_dict={
        model['placeholders']['cost_value']: cost}), m)
    writers[name].flush()
    return cost, acc


def define_input_stream(model):
    model['streams'] = dict()
    with tf.name_scope('input'):
        model['placeholders']['iterator_handle'] = tf.placeholder(
            tf.string, shape=[])
        tf.logging.info('input placeholder name: %s',
                        model['placeholders']['iterator_handle'])
        model['streams']['iterator'] = tf.data.Iterator.from_string_handle(
            model['placeholders']['iterator_handle'],
            model['data']['train']['dataset'].output_types,
            model['data']['train']['dataset'].output_shapes)
        model['streams']['images'], model['streams']['labels'] = model[
            'streams']['iterator'].get_next()


def define_data_iterators(data):
    data['train']['iterator'] = data['train'][
        'dataset'].make_initializable_iterator()
    data['test']['iterator'] = data['test'][
        'dataset'].make_initializable_iterator()
    data['hessian']['iterator'] = data['hessian'][
        'dataset'].make_initializable_iterator()


def define_model(params,
                 fake=False,
                 data=None,
                 streams=None,
                 placeholders=None,
                 ops=None,
                 graph=None,
                 saver=None):
    model_container = dict()
    if data is None:
        graph = tf.Graph()
        with graph.as_default():
            model_container['data'] = get_weakened_data(params, fake)
            model_container['placeholders'] = dict()
            define_input_stream(model_container)
            define_data_iterators(model_container['data'])
            neural_network_model(model_container, params)
            model_container['saver'] = tf.train.Saver()
            model_container['graph'] = graph
    else:
        model_container['placeholders'] = placeholders
        model_container['streams'] = streams
        model_container['data'] = data
        model_container['ops'] = ops
        model_container['graph'] = graph
        model_container['saver'] = saver

    return model_container


def setup_log_writers(sess, sess_name, log_dir=constants.LOG_DIR):
    writers = dict()
    writers['train'] = tf.summary.FileWriter(
        log_dir + '/' + sess_name + '/train', sess.graph)
    writers['test'] = tf.summary.FileWriter(
        log_dir + '/' + sess_name + '/test')
    return writers


def initialize_data_iterators(model, sess):
    model['data']['train']['handle'] = sess.run(
        model['data']['train']['iterator'].string_handle())
    model['data']['test']['handle'] = sess.run(
        model['data']['test']['iterator'].string_handle())
    model['data']['hessian']['handle'] = sess.run(
        model['data']['hessian']['iterator'].string_handle())


def initialize_session(model, sess):
    initialize_data_iterators(model, sess)
    sess.run(tf.global_variables_initializer())


def train_neural_network(model,
                         sess_name,
                         params,
                         res,
                         eval_tests=True,
                         eval_every_n=constants.EVAL_EVERY_N,
                         model_save_dir=constants.MODEL_SAVE_DIR,
                         calc_results=True,
                         sess=None,
                         sess_opt={'config': constants.SESSION_CONFIG},
                         writers=None):
    if sess is None:
        process = psutil.Process(os.getpid())
        tf.logging.info('Memory used before starting session: %s GB',
                        process.memory_info().rss / 1e9)
        with tf.Session(graph=model['graph'], **sess_opt) as sess:
            tf.logging.info('Memory used before initializing session: %s GB',
                            process.memory_info().rss / 1e9)
            initialize_session(model, sess)
            tf.logging.info('Memory used before training session: %s GB',
                            process.memory_info().rss / 1e9)
            train_neural_network_(model, sess_name, params, res, sess,
                                  eval_tests, eval_every_n, model_save_dir, calc_results, writers)
            tf.logging.info('Memory used after training with session: %s GB',
                            process.memory_info().rss / 1e9)
            clear_session(sess)
            tf.logging.info('Memory used after clearing session: %s GB',
                            process.memory_info().rss / 1e9)
        tf.logging.info('Memory used after ending session: %s GB',
                        process.memory_info().rss / 1e9)
    else:
        train_neural_network_(model, sess_name, params, res, sess, eval_tests,
                              eval_every_n, model_save_dir, calc_results, writers)


def train_neural_network_(model,
                          sess_name,
                          params,
                          res,
                          sess,
                          eval_tests,
                          eval_every_n,
                          model_save_dir,
                          calc_results,
                          writers):
    process = psutil.Process(os.getpid())
    if writers is None:
        writers = setup_log_writers(sess, sess_name)
    gc.collect()
    train_acc = 0
    test_acc = 0
    start_time = time.time()
    model_checkpoint_path = os.path.join(model_save_dir, sess_name)
    checkpoint_pattern = os.path.join(model_checkpoint_path, 'model.ckpt')
    tf.gfile.MakeDirs(model_checkpoint_path)

    try:
        M = params['M']
        patience = 10
        train_costs = np.ones(patience) * np.inf
        test_cost_min = np.inf
        tf.logging.info('Starting training')
        tf.logging.info('Memory used at training start: %s GB',
                        process.memory_info().rss / 1e9)
        for m in range(0, M):
            sess.run(model['data']['train']['iterator'].initializer)
            n = 0
            tf.logging.info('Epoch: %s', m)
            tf.logging.info('Memory used at epoch %s: %s GB', m,
                            process.memory_info().rss / 1e9)
            # batch training
            while True:
                n += 1
                try:
                    train_cost, _ = sess.run(
                        [model['ops']['cost'], model['ops']['train']],
                        feed_dict={
                            model['placeholders']['iterator_handle']:
                            model['data']['train']['handle']
                        })
                except tf.errors.OutOfRangeError:
                    break

            # keep a running record of the val(patience) last costs
            update_cost_record(train_cost, train_costs)
            tf.logging.info('Train cost and accuracy at step %s: %s, %s', m,
                            train_cost, train_acc)
            if ((m % eval_every_n) == 0) and eval_tests:
                tf.logging.info('Memory used before eval step: %s GB',
                                process.memory_info().rss / 1e9)
                test_cost, test_acc = eval_scores(
                    m, model, writers, 'test', sess, verbose=False)
                train_cost, train_acc = eval_scores(m, model, writers,
                                                    'train', sess, verbose=False)
                if test_cost_min > test_cost:
                    test_cost_min = test_cost
                    save_path = model['saver'].save(sess, checkpoint_pattern)
                    tf.logging.info('Saved model to %s', save_path)
                if early_stopping(train_costs, test_cost, test_cost_min):
                    tf.logging.info(
                        'Early stopping, train: cost: %g, accuracy: %g, test: cost: %g, accuracy: %g',
                        train_cost, train_acc, test_cost, test_acc)
                    break

                tf.logging.info('Test accuracy at step %s: %s', m, test_acc)
                tf.logging.info('Memory used after eval step: %s GB',
                                process.memory_info().rss / 1e9)
        gc.collect()
        tf.logging.info('Training Completed')
    finally:
        writers['train'].close()
        writers['test'].close()
        tf.logging.info('Writers closed')
    gc.collect()
    tf.logging.info('Restoring last checkpoint')
    ckpt = tf.train.get_checkpoint_state(model_checkpoint_path)
    tf.logging.info('ckpt: %s', ckpt)
    if ckpt:
        model['saver'].restore(sess, ckpt.model_checkpoint_path)
        tf.logging.info('Restored checkpoint')
    else:
        tf.logging.error('No checkpoint found')

    graph_def = model['graph'].as_graph_def()
    regex = re.compile('^[^:]*')
    ops_vars_names = [re.match(regex, var.name).group(0)
                      for var in tf.trainable_variables()]
    ops_vars_names.extend(
        [model['ops']['hessian'].name[:-2], model['ops']['hess_eigs'].name[:-2]])
    tf.logging.info('ops_vars_names: %s', ops_vars_names)
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph_def, ops_vars_names)
    with tf.gfile.GFile(constants.FROZEN_GRAPH, "wb") as f:
        f.write(frozen_graph_def.SerializeToString())

    if calc_results:
        calc_final_results(res, model, m, start_time, params,
                           sess)


def calc_hess_results(res, model, nodes_to_preserve, model_checkpoint_path, params, sess_name):
    process = psutil.Process(os.getpid())
    # -----  HESSIAN CALCULATION -----

    tf.logging.info('Memory used at start of calc_hess_results: %s GB',
                    process.memory_info().rss / 1e9)

    model['graph'] = tf.Graph()
    model['ops'] = dict()
    model['placeholders'] = dict()
    tf.reset_default_graph()
    gc.collect()
    tf.logging.info('Memory used after model deletion: %s GB',
                    process.memory_info().rss / 1e9)

    with open(constants.FROZEN_GRAPH, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    tf.logging.info('Memory used after graph_def read: %s GB',
                    process.memory_info().rss / 1e9)

    # Keep only the necessary parts of the graph to preserve memory
    with model['graph'].as_default():
        tf.graph_util.remove_training_nodes(
            graph_def, protected_nodes=nodes_to_preserve)
        model['ops']['hessian'], model['ops']['hess_eigs'], model['placeholders']['iterator_handle'], model['placeholders']['hessian'] = tf.import_graph_def(
            graph_def, return_elements=nodes_to_preserve)
        # add the iterators again
        model['data'] = get_weakened_data(params)
        define_data_iterators(model['data'])
        tf.logging.info('Memory used after graph_def import: %s GB',
                        process.memory_info().rss / 1e9)

        with tf.Session(graph=model['graph'], config=constants.SESSION_CONFIG) as sess:
            # writer = tf.summary.FileWriter(
            #    constants.LOG_DIR + '/' + sess_name + '/hessian', sess.graph)
            #ckpt = tf.train.get_checkpoint_state(model_checkpoint_path)
            #model['saver'].restore(sess, ckpt.model_checkpoint_path)
            initialize_session(model, sess)
            tf.logging.info('Calculating Hessian')
            tf.logging.info('Memory used before Hessian calculation: %s GB',
                            process.memory_info().rss / 1e9)
            tf.logging.info('hessian op: %s', model['ops']['hessian'])
            hessian = sum_over_dataset('hessian', model, model['ops']['hessian'], sess,
                                       constants.MAX_BATCHES_TO_SAMPLE_HESS, writer=None, collect_metadata=False)
            tf.logging.info('Calculating eigenvalues of the Hessian')
            tf.logging.info('Memory used before Hessian eigenvalue calculation: %s GB',
                            process.memory_info().rss / 1e9)
            hess_eig = sess.run(model['ops']['hess_eigs'],
                                feed_dict={model['placeholders']['hessian']: hessian})
            tf.logging.info('Memory used after Hessian eigenvalue calculation: %s GB',
                            process.memory_info().rss / 1e9)
            tf.logging.info('Completed the eigenvalue calculation')
            tf.logging.info('Memory currently used: %s GB',
                            process.memory_info().rss / 1e9)
    gc.collect()
    tf.logging.info('Memory used after hess session: %s GB',
                    process.memory_info().rss / 1e9)
    # -------------------------------------------
    # ----- DELETE HESSIAN TO CLEAR MEMORY ------
    del hessian
    gc.collect()
    # -------------------------------------------
    # -------------------------------------------

    tf.logging.info('Memory used after deletion of Hessian: %s GB',
                    process.memory_info().rss / 1e9)

    hess_eig /= min(constants.MAX_BATCHES_TO_SAMPLE_HESS,
                    model['data']['hessian']['num_batches'])

    hess_eig.sort()
    np.savetxt('hess_eig_' + sess_name + '.out', hess_eig, delimiter=',')

    log_top_fifty_eigs = np.log10(hess_eig[-50:])
    top_log_eig_sum = np.sum(log_top_fifty_eigs)
    res['eigenvalues'].append(hess_eig)
    res['log_top_fifty_eigs'].append(log_top_fifty_eigs)
    res['top_log_eig_sum'].append(top_log_eig_sum)
    curr_min = min(log_top_fifty_eigs)
    curr_max = max(log_top_fifty_eigs)
    if (res['max_log_eig']) < curr_max:
        res['max_log_eig'] = curr_max
    if (res['min_log_eig']) > curr_min:
        res['min_log_eig'] = curr_min


def calc_final_results(res, model, m, start_time, params, sess):
    data = model['data']
    ops = model['ops']
    tf.logging.info('Calculating final training results')
    train_acc = sum_over_dataset(
        'train', model, ops['accuracy'], sess) / data['train']['num_batches']

    tf.logging.info('Calculating final test results')

    test_acc = sum_over_dataset(
        'test', model, ops['accuracy'], sess) / data['test']['num_batches']

    tf.logging.info('Calculating plotting variables')

    res['test_acc'].append(test_acc * 100)
    res['noise_mix'].append(params['noise_mix'])
    res['train_acc'].append(train_acc * 100)
    res['exec_time'].append(time.time() - start_time)
    res['M'].append(m + 1)

    tf.logging.info('Results: %s', res)
