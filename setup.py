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
        'av',
    ]
)