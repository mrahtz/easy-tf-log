import os
import time

import tensorflow as tf
from tensorflow.core.util import event_pb2
from tensorflow.python.summary.writer.event_file_writer import EventFileWriter


class Logger(object):
    DEFAULT = None

    def __init__(self):
        self.key_steps = {}

    def set_log_dir(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = EventFileWriter(log_dir)

    def set_writer(self, writer):
        self.writer = writer

    def logkv(self, k, v):
        def summary_val(k, v):
            kwargs = {'tag': k, 'simple_value': float(v)}
            return tf.Summary.Value(**kwargs)

        summary = tf.Summary(value=[summary_val(k, v)])
        event = event_pb2.Event(wall_time=time.time(), summary=summary)
        # Use a separate step counter for each key
        if k not in self.key_steps:
            self.key_steps[k] = 1
        event.step = self.key_steps[k]
        self.writer.add_event(event)
        self.writer.flush()
        self.key_steps[k] += 1

    def close(self):
        if self.writer:
            self.writer.Close()
            self.writer = None


def set_dir(log_dir):
    Logger.DEFAULT = Logger()
    Logger.DEFAULT.set_log_dir(log_dir)


def set_writer(writer):
    Logger.DEFAULT = Logger()
    Logger.DEFAULT.set_writer(writer)


def tflog(key, value):
    if not Logger.DEFAULT:
        set_dir('logs')
    Logger.DEFAULT.logkv(key, value)
