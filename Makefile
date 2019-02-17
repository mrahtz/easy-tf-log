dist: easy_tf_log.py
	rm -rf dist
	python3 setup.py sdist bdist_wheel
	twine upload dist/*
