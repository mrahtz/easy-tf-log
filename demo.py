#!/usr/bin/env python
import time

import easy_tf_log

# Logging using the global logger

# Will log to automatically-created 'logs' directory
for i in range(10):
    easy_tf_log.tflog('foo', i)
for j in range(10, 20):
    easy_tf_log.tflog('bar', j)

easy_tf_log.set_dir('logs2')

for k in range(20, 30):
    easy_tf_log.tflog('baz', k)
for l in range(5):
    easy_tf_log.tflog('qux', l, step=(10 * l))

# Logging using a Logger object

logger = easy_tf_log.Logger(log_dir='logs3')

for i in range(10):
    logger.log_key_value('quux', i)

logger.log_list_stats('quuz', [1, 2, 3, 4, 5])

logger.measure_rate('corge', 10)
time.sleep(1)
logger.measure_rate('corge', 20)  # Logged rate: (20 - 10) / 1
time.sleep(2)
logger.measure_rate('corge', 30)  # Logged rate: (30 - 20) / 2
