# Easy TensorFlow Logging

Are you prototyping something and want to be able to _magically_ graph some value
without going through all the usual steps to set up TensorFlow logging properly?

`easy-tf-log` is a simple module to do just that.

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

Based on logging code from OpenAI's [baselines](https://github.com/openai/baselines).

## Installation

`pip install easy-tf-log`

Note that TensorFlow must be installed separately.

## Usage

By default, `easy-tf-log` saves event files to a directory `logs`.
To change the directory, call `easy_tf_log.set_dir(log_dir)`.

`easy-tf-log` also supports writing using an existing `EventFileWriter` created
by e.g. an instance of `tf.summary.FileWriter`: call
`easy_tf_log.set_writer(file_writer.event_writer)`. (However, not that because
`EventsFileWriter` uses a sub-thread to write events, this is not fork-safe. If
you set this in one process and then try to use `easy-tf-log` a child process,
it will hang.)

To log a value, use `tflog(key, value)`.

See [`demo.py`](demo.py) for a full demo.

## Tests

[![CircleCI](https://circleci.com/gh/mrahtz/easy-tf-log/tree/master.svg?style=svg&circle-token=4750ebc3733b859421a6453d2fe15c363480fa1c)](https://circleci.com/gh/mrahtz/easy-tf-log/tree/master)
