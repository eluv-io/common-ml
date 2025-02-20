from setuptools import setup, find_packages

setup(
    name="common-ml",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'ffmpeg-python==0.2.0',
        'numpy',
        'ujson',
        'loguru',
        'schema',
        'PyYAML',
        'marshmallow',
        'pillow',
        'opencv-python',
        'typing-extensions',
        'quick_test_py @ git+https://github.com/elv-nickB/quick_test_py.git#egg=quick_test_py',
        'elv-client-py @ git+https://github.com/eluv-io/elv-client-py.git#egg=elv-client-py',
    ]
)