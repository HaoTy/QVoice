from setuptools import setup, find_packages

setup(
    name='qvoice',
    version='0.1',
    description='QVoice: Quantum Variational Optimization with In-Constraint Probability',
    author='Tianyi Hao',
    url='https://github.com/haoty/QVoice',
    packages=find_packages(exclude=['test*', 'scripts', 'assets', 'notebooks', 'doc']),
    python_requires='>=3.10',
    install_requires=[
        'numpy>=1.22.4',
        'qiskit>=0.37.1',
        'qiskit_finance>=0.3.3',
        'qiskit_optimization>=0.4.0',
        'networkx>=2.8.4',
        'matplotlib>=3.5.2'
    ],
    # test_suite='test.test',
)
