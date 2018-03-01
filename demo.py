import easy_tf_log
from easy_tf_log import logkv

for i in range(10):
    logkv('foo', i)
for j in range(10, 20):
    logkv('bar', j)

easy_tf_log.set_dir('logs2')

for k in range(20, 30):
    logkv('baz', k)
