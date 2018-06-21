import os.path as path

from setuptools import setup

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='easy_tf_log',
    version='1.7',
    description='TensorFlow logging made easy',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mrahtz/easy-tf-log',
    author='Matthew Rahtz',
    author_email='matthew.rahtz@gmail.com',
    keywords='tensorflow graph graphs graphing',
    py_modules=['easy_tf_log']
)
