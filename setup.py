from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='nflows-tools',
    version='0.1',
    packages=['nflows_tools'],
    entry_points={
        'console_scripts': [
            'nflows_validate_offsets=nflows_tools.validate_offsets:main',
            'nflows_validate_output=nflows_tools.validate_output:main',
            'nflows_aggreg_output=nflows_tools.aggreg:main',
            'nflows_generate_config=nflows_tools.config:main',
            'nflows_generate_dot=nflows_tools.dot:main',
            'nflows_generate_gantt=nflows_tools.gantt:main',
            'nflows_generate_profile=nflows_tools.profile:main',
            'nflows_generate_slurm=nflows_tools.slurm:main',
            'nflows_find_workflow_structs=nflows_tools.structures:main',
        ],
    },
    install_requires=requirements,
    author='Aurelio Vivas',
    description='Command-line utilities for validating nFlows output and analyzing execution results',
)
