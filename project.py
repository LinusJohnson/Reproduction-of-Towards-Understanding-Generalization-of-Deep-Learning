import tensorflow as tf
import numpy as np
import constants
import functions as funcs
import gc
from memory_profiler import profile
from multiprocessing import Process, Queue
import os
import psutil

# ------- MAIN FUNCTION ------- #


def run_tf_train_containment_process(queue, params, sess_name):
    process = psutil.Process(os.getpid())
    tf.logging.info('memory used at start of tf_train_containment_process: %s GB',
                    process.memory_info().rss / 1e9)
    model_res = funcs.initialize_model_res_dict()
    model = funcs.define_model(params, fake=True)
    tf.logging.info('memory used after defining the model in tf_train_containment_process: %s GB',
                    process.memory_info().rss / 1e9)
    funcs.train_neural_network(model, sess_name, params, model_res)
    tf.logging.info('memory used after training in tf_train_containment_process: %s GB',
                    process.memory_info().rss / 1e9)
    queue.put(model_res)
    nodes_to_preserve = [model['ops']['hessian'].name,
                         model['ops']['hess_eigs'].name, model['placeholders']['iterator_handle'].name, model['placeholders']['hessian'].name]
    queue.put(nodes_to_preserve)
    model.clear()
    del model
    gc.collect()
    tf.reset_default_graph()
    tf.logging.info('memory used at end of tf_train_containment_process: %s GB',
                    process.memory_info().rss / 1e9)


def run_tf_hessian_containment_process(queue, nodes_to_preserve, model_res, params, sess_name):
    process = psutil.Process(os.getpid())
    tf.logging.info('memory used at start of tf_hessian_containment_process: %s GB',
                    process.memory_info().rss / 1e9)
    with tf.device('/device:GPU:0'):
        model = dict()
        funcs.calc_hess_results(model_res, model, nodes_to_preserve, os.path.join(
            constants.MODEL_SAVE_DIR, sess_name), params, sess_name)
        tf.logging.info('memory used after calculating the Hessian in tf_hessian_containment_process: %s GB',
                        process.memory_info().rss / 1e9)
        queue.put(model_res)
        model.clear()
        del model
        gc.collect()
        tf.reset_default_graph()
        tf.logging.info('memory used at end of tf_hessian_containment_process: %s GB',
                        process.memory_info().rss / 1e9)


def main():
    funcs.setup_logging()
    process = psutil.Process(os.getpid())
    model_res = funcs.initialize_model_res_dict()
    plots = funcs.create_plots()
    results = funcs.init_res_dict()
    noise_mix_vals = [param['noise_mix']
                      for param in constants.PARAMS_TO_EVALUATE.values()]
    c_scale = 1/(max(noise_mix_vals)-min(noise_mix_vals))
    for prob in constants.PARAMS_TO_EVALUATE.keys():
        tf.logging.info(prob)
        params = funcs.get_params(prob)
        tf.logging.info(params)
        sess_name = funcs.get_session_name(prob, 0)
        results_queue = Queue()
        train_proc = Process(target=run_tf_train_containment_process,
                             args=(results_queue, params, sess_name))
        train_proc.start()
        model_res = results_queue.get()
        nodes_to_preserve = results_queue.get()
        funcs.optimize_frozen_graph(nodes_to_preserve)
        train_proc.join()
        tf.logging.info('memory used after train_proc.join: %s GB',
                        process.memory_info().rss / 1e9)
        hess_proc = Process(target=run_tf_hessian_containment_process,
                            args=(results_queue, nodes_to_preserve, model_res, params, sess_name))
        hess_proc.start()
        model_res = results_queue.get()
        hess_proc.join()
        tf.logging.info('memory used after hess_proc.join: %s GB',
                        process.memory_info().rss / 1e9)
        gc.collect()
        funcs.update_plots(model_res, params,
                           sess_name, plots, c_scale=c_scale)
        funcs.add_sample_to_res(model_res, prob, 0, results)
        funcs.calc_stats(results, prob)


if __name__ == '__main__':
    main()
