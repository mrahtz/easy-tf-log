import importlib
import os
import os.path as osp
import queue
import tempfile
import time
import unittest
from multiprocessing import Queue, Process

import numpy as np
import tensorflow as tf

import easy_tf_log


class TestEasyTFLog(unittest.TestCase):

    def setUp(self):
        importlib.reload(easy_tf_log)
        print(self._testMethodName)

    def test_no_setup(self):
        """
        Test that if tflog() is used without any extra setup, a directory
        'logs' is created in the current directory containing the event file.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            easy_tf_log.tflog('var', 0)
            self.assertEqual(os.listdir(), ['logs'])
            self.assertIn('events.out.tfevents', os.listdir('logs')[0])

    def test_set_dir(self):
        """
        Confirm that set_dir works.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            easy_tf_log.set_dir('logs2')
            easy_tf_log.tflog('var', 0)
            self.assertEqual(os.listdir(), ['logs2'])
            self.assertIn('events.out.tfevents', os.listdir('logs2')[0])

    def test_set_writer(self):
        """
        Check that when using an EventFileWriter from a FileWriter,
        the resulting events file contains events from both the FileWriter
        and easy_tf_log.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            writer = tf.summary.FileWriter('logs')

            var = tf.Variable(0.0)
            summary_op = tf.summary.scalar('tf_var', var)
            sess = tf.Session()
            sess.run(var.initializer)
            summary = sess.run(summary_op)
            writer.add_summary(summary)

            easy_tf_log.set_writer(writer.event_writer)
            easy_tf_log.tflog('easy-tf-log_var', 0)

            self.assertEqual(os.listdir(), ['logs'])
            event_filename = osp.join('logs', os.listdir('logs')[0])
            self.assertIn('events.out.tfevents', event_filename)

            tags = set()
            for event in tf.train.summary_iterator(event_filename):
                for value in event.summary.value:
                    tags.add(value.tag)
            self.assertIn('tf_var', tags)
            self.assertIn('easy-tf-log_var', tags)

    def test_full(self):
        """
        Log a few values and check that the event file contain the expected
        values.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            for i in range(10):
                easy_tf_log.tflog('foo', i)
            for i in range(10):
                easy_tf_log.tflog('bar', i)

            event_filename = osp.join('logs', os.listdir('logs')[0])
            event_n = 0
            for event in tf.train.summary_iterator(event_filename):
                if event_n == 0:  # metadata
                    event_n += 1
                    continue
                if event_n <= 10:
                    self.assertEqual(event.step, event_n - 1)
                    self.assertEqual(event.summary.value[0].tag, "foo")
                    self.assertEqual(event.summary.value[0].simple_value,
                                     float(event_n - 1))
                if event_n > 10 and event_n <= 20:
                    self.assertEqual(event.step, event_n - 10 - 1)
                    self.assertEqual(event.summary.value[0].tag, "bar")
                    self.assertEqual(event.summary.value[0].simple_value,
                                     float(event_n - 10 - 1))
                event_n += 1

    def test_explicit_step(self):
        """
        Log a few values explicitly setting the step number.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            for i in range(5):
                easy_tf_log.tflog('foo', i, step=(10 * i))
            # These ones should continue from where the previous ones left off
            for i in range(5):
                easy_tf_log.tflog('foo', i)

            event_filename = osp.join('logs', os.listdir('logs')[0])
            event_n = 0
            for event in tf.train.summary_iterator(event_filename):
                if event_n == 0:  # metadata
                    event_n += 1
                    continue
                if event_n <= 5:
                    self.assertEqual(event.step, 10 * (event_n - 1))
                if event_n > 5 and event_n <= 10:
                    self.assertEqual(event.step, 40 + (event_n - 5))
                event_n += 1

    def test_fork(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            easy_tf_log.set_dir(temp_dir)

            def f(queue):
                easy_tf_log.tflog('foo', 0)
                queue.put(True)

            q = Queue()
            Process(target=f, args=[q], daemon=True).start()
            try:
                q.get(timeout=1.0)
            except queue.Empty:
                self.fail("Process did not return")

    def test_measure_rate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = easy_tf_log.Logger(log_dir=temp_dir)

            logger.measure_rate('foo', 0)
            time.sleep(1)
            logger.measure_rate('foo', 10)
            time.sleep(1)
            logger.measure_rate('foo', 25)

            event_filename = list(os.scandir(temp_dir))[0].path
            event_n = 0
            rates = []
            for event in tf.train.summary_iterator(event_filename):
                if event_n == 0:  # metadata
                    event_n += 1
                    continue
                rates.append(event.summary.value[0].simple_value)
                event_n += 1
            np.testing.assert_array_almost_equal(rates, [10., 15.], decimal=1)


if __name__ == '__main__':
    unittest.main()
