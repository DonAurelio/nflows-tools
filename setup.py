from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='nflows-tools',
    version='1.0.0',
    packages=['nflows_tools'],
    entry_points={
        'console_scripts': [
            'nflows_validate_profile_offsets=nflows_tools.validation_profile_offsets:main',
            'nflows_validate_profile_output=nflows_tools.validation_profile_output:main',

            'nflows_experiment_config=nflows_tools.experiment_config:main',
            'nflows_experiment_slurm=nflows_tools.experiment_slurm:main',
            'nflows_experiment_collect=nflows_tools.experiment_collect:main',

            'nflows_workflow_wfformat=nflows_tools.workflow_wfformat:main',
            'nflows_workflow_structures=nflows_tools.workflow_structures:main',

            'nflows_profile_compact=nflows_tools.profile_compact:main',
            'nflows_profile_graph=nflows_tools.profile_graph:main',
        ],
    },
    install_requires=requirements,
    author='Aurelio Vivas',
    description='Command-line utilities for validating nFlows output and analyzing execution results',
)
