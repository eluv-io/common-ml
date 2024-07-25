from setuptools import setup

setup(
    name="common-ml",
    version="0.1",
    packages=['common_ml'],
    install_requires=[
        'opencv-python',
        'ffmpeg-python==0.2.0',
        'ujson==5.7.0',
        'loguru==0.5.2',
        'schema==0.7.5',
        'PyYAML==6.0.1',
        'marshmallow==3.19.0',
    ]
)