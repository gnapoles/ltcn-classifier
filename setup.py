from setuptools import setup

setup(
    name='ltcnclassifier',
    version='0.1.0',
    description='Long-term Cognitive Networks for pattern classification',
    long_description="This package introduces the Long-Term Cognitive Network (LTCN) model for structured pattern " +
                     "classification problems. This recurrent neural network incorporates a quasi-nonlinear reasoning " +
                     "rule that allows controlling the ammout of nonlinearity in the reasoning mechanism. Furthermore, " +
                     "this neural classifier uses a recurrence-aware decision model that evades the issues posed by " +
                     "the unique fixed point while introducing a deterministic learning algorithm to compute the " +
                     "tunable parameters. The experiments in the original paper show that this classifier obtains " +
                     "competitive results when compared to state-of-the-art white and black-box models.",
    url='https://github.com/gnapoles/ltcn-classifier',
    author='Gonzalo Nápoles',
    author_email='g.r.napoles@uvt.nl',
    license='MIT License',
    packages=['ltcnclassifier'],
    install_requires=['numpy', 'scipy', 'scikit-learn', 'pandas'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.8',
    ],
)
