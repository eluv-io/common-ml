from setuptools import setup

setup(
    name="common-ml",
    version="0.1",
    packages=['common_ml'],
    install_requires=[
        'opencv-python',
        'ffmpeg-python==0.2.0',
        'ujson',
        'loguru',
        'schema',
        'PyYAML',
        'marshmallow',
    ]
)