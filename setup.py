try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

config = {
    'description': 'wiens',
    'author': 'Neil Seward',
    'author_email': 'neil.seawrd@uoit.ca',
    'version': '0.0.1',
    'packages': find_packages(),
    'name': 'wiens'
}

setup(**config)
