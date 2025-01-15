from setuptools import setup

setup(
    name="common-ml",
    version="0.1",
    packages=['common_ml'],
    install_requires=[
        'ffmpeg-python==0.2.0',
        'numpy',
        'ujson',
        'loguru',
        'schema',
        'PyYAML',
        'marshmallow',
        'pillow',
        'quick_test_py @ git+https://github.com/elv-nickB/quick_test_py.git#egg=quick_test_py',
        'elv-client-py @ git+https://github.com/eluv-io/elv-client-py.git#egg=elv-client-py',
    ]
)