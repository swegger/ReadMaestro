from distutils.core import setup
import numpy as np



setup(name='ReadMaestro',
      version='1.0',
      description='Basic timeseries organization of neural data.',
      author='Nathan Hall',
      author_email='nathan.hall@duke.edu',
      url='https://',
      packages=['ReadMaestro'],
      include_dirs=[np.get_include()],
     )
