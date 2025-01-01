from setuptools import setup, find_packages

setup(
    name='soed',
    version='1.0.5',
    packages=find_packages(),
    description='Self Organizing Error Driven Artificial Neural Network',
    long_description=open('Readme.md').read(),
    long_description_content_type='text/markdown',
    author='Roy Jafari',
    author_email='royxjafari@gmail.com',
    maintainer="Jason Jafari",
    maintainer_email="me@jasonjafari.com",
    url='https://github.com/royjafari/SOED',  # URL to your package repository
    classifiers=[
       'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',  # Correct classifier for the MIT License
        'Programming Language :: Python :: 3.12',
    ],
    keywords='machine learning AI Artificial Neural Network',  # Keywords for your package
    install_requires=[
        'numpy==2.2.1',
        'pandas==2.2.3',
        'scikit-learn==1.6.0',
        'MiniSom==2.3.3',
        "scipy==1.14.1",
        "python-dateutil==2.9.0.post0",
        "six==1.17.0",
        "threadpoolctl==3.5.0",
        "tzdata==2024.2"
    ],
    project_urls={  # Optional
        'Documentation': 'https://github.com/royjafari/SOED/blob/main/README.md',
        'Source': 'https://github.com/royjafari/SOED',
        'Tracker': 'https://github.com/royjafari/SOED/issues',
    },
)
