from setuptools import setup

setup(
    name='EOLES',
    version='2.0',
    packages=['eoles', 'eoles.inputs', 'eoles.inputs.config', 'eoles.inputs.demand', 'eoles.inputs.historical_data',
              'eoles.inputs.hourly_profiles', 'eoles.inputs.technology_characteristics',
              'eoles.inputs/tecnology_potential'],
    url='',
    license='',
    author='Célia Escribe',
    author_email='celia.escribe@polytechnique.edu',
    description='Generation capacity expansion model'
)
