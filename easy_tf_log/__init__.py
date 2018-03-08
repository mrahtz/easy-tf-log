import os
import os.path as osp
import time

import tensorflow as tf
from tensorflow.core.util import event_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.util import compat


class Logger(object):
    DEFAULT = None

    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.dir = log_dir
        self.key_steps = {}
        prefix = 'events'
        path = osp.join(osp.abspath(log_dir), prefix)
        self.event_pb2 = event_pb2
        self.pywrap_tensorflow = pywrap_tensorflow
        self.writer = pywrap_tensorflow.EventsWriter(compat.as_bytes(path))

    def logkv(self, k, v):
        def summary_val(k, v):
            kwargs = {'tag': k, 'simple_value': float(v)}
            return tf.Summary.Value(**kwargs)
        summary = tf.Summary(value=[summary_val(k, v)])
        event = self.event_pb2.Event(wall_time=time.time(), summary=summary)
        if k not in self.key_steps:
            self.key_steps[k] = 1
        event.step = self.key_steps[k]
        self.writer.WriteEvent(event)
        self.writer.Flush()
        self.key_steps[k] += 1

    def close(self):
        if self.writer:
            self.writer.Close()
            self.writer = None


Logger.DEFAULT = Logger(log_dir='logs')


def set_dir(log_dir):
    Logger.DEFAULT = Logger(log_dir)


def logkv(key, val):
    Logger.DEFAULT.logkv(key, val)
