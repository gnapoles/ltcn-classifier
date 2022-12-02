from setuptools import setup

setup(
    name='ltcnclassifier',
    version='0.1.0',
    description='Long-term Cognitive Networks for pattern classification',
    long_description="Long-term Cognitive Networks are trained with an inverse learning rule. " +
                     "In this model, the weights connecting the input neurons are coefficients " +
                     "of multiple regressions models while the weights connecting the temporal " +
                     "states with and outputs are computed using a learning method (the Moore–Penrose " +
                     "inverse method when no regularization is needed or the Ridge regression method " +
                     "when the model might overfit the data).",
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
