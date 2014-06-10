from distutils.core import setup

setup(
    name='akx',
    version='0.95a',

    packages=[
        'akx',
    ],

    package_data={
        'akx': ['defaults.cfg'],
    },

    install_requires=[
        'ctree',
    ]
)

