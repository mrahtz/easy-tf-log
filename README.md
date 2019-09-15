# Easy TensorFlow Logging

[![CircleCI](https://circleci.com/gh/mrahtz/easy-tf-log.svg?style=shield)](https://circleci.com/gh/mrahtz/easy-tf-log)

Are you prototyping something and want to be able to _magically_ graph some value
without going through all the usual steps to set up TensorFlow logging properly?

`easy_tf_log` is a simple module to do just that.

```
from easy_tf_log import tflog
```

then you can do

```
for i in range(10):
    tflog('really_interesting_variable_name', i)
```

and you'll find a directory `logs` that you can point TensorBoard to

`$ tensorboard --logdir logs`

to get

![](https://github.com/mrahtz/easy-tf-log/blob/master/tensorboard_screenshot.png)

See [`demo.py`](demo.py) for a full demo.

Based on logging code from OpenAI's [baselines](https://github.com/openai/baselines).

## Installation

`pip install easy-tf-log`

Note that TensorFlow must be installed separately.

## Documentation

`easy-tf-log` supports logging using either a global logger or an instantiated logger object.

The global logger is good for very quick prototypes, but for anything more complicated,
you'll probably want to instantiate your own `Logger` object.

### Global logger

* `easy_tf_log.tflog(key, value, step=None)`
  * Logs `value` (int or float) under the name `key` (string).
  * `step` (int) sets the step associated with `value` explicitly.
    If not specified, the step will increment on each call.
* `easy_tf_log.set_dir(log_dir)`
  * Sets the global logger to log to the specified directory.
  * `log_dir` can be an absolute or a relative path.
* `easy_tf_log.set_writer(writer)`
  * Sets the global logger to log using the specified `tf.summary.FileWriter` instance.
  
By default (i.e. if `set_dir` is not called), the global logger logs to a `logs` directory
automatically created in the working directory.

### Logger object

* `logger = easy_tf_log.Logger(log_dir=None, writer=None)`
  * Create a `Logger`.
  * `log_dir`: an absolute of relative path specifying the directory to log to.
  * `writer`: an existing `tf.summary.FileWriter` instance to use for logging.
  * If neither `log_dir` nor `writer` are specified, the logger will log to a `logs` directory in the
    working directory. If both are specified, the constructor will raise a `ValueError`.
* `logger.log_key_value(key, value, step=None)`
  * See `tflog`.
* `logger.log_list_stats(key, values_list)`
  * Log the minimum, maximum, mean, and standard deviation of `values_list` (a list of ints or floats).
* `logger.measure_rate(key, value)`
  * Log the rate at which `value` (int or float) changes per second.
  * The first call internally stores the time of the first value;
    the second call logs the change between the second value and the first value divided by the
    time between the calls; etc.
* `logger.set_dir(log_dir)`
  * See `easy_tf_log.set_dir(log_dir)`.
* `logger.set_writer(writer)`
  * See `easy_tf_log.set_writer(writer)`.
* `logger.close()`
  * Flush logs and close the log file handle.
