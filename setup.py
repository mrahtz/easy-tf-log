from setuptools import setup

setup(name='easy_tf_log',
    version='1.1',
    py_modules=['easy_tf_log'],
    extras_require={
        'tf': ['tensorflow'],
        'tf_gpu': ['tensorflow-gpu'],
    }
)
