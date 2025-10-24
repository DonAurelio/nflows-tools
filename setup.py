from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='nflows-tools',
    version='0.1',
    packages=['nflows_tools'],
    entry_points={
        'console_scripts': [
            'nflows_validate_profile_offsets=nflows_tools.validation.profile_offsets:main',
            'nflows_validate_profile_output=nflows_tools.validation.profile_output:main',

            'nflows_expriment_config=nflows_tools.experiment.config:main',
            'nflows_experiment_slurm=nflows_tools.experiment.slurm:main',
            'nflows_experiment_collect=nflows_tools.experiment.collect:main',

            'nflows_workflow_wfformat=nflows_tools.workflow.wfformat:main',
            'nflows_workflow_structures=nflows_tools.workflow.structures:main',

            'nflows_profile_compact=nflows_tools.profile.compact:main',
            'nflows_profile_graph=nflows_tools.profile.graph:main',
        ],
    },
    install_requires=requirements,
    author='Aurelio Vivas',
    description='Command-line utilities for validating nFlows output and analyzing execution results',
)
