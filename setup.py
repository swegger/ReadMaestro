from distutils.core import setup
import numpy as np



setup(name='ReadMaestro',
      version='1.0',
      description='Basic timeseries organization of neural data.',
      author='Seth Egger',
      author_email='sethegger@gmail.com',
      url='https://',
      packages=['ReadMaestro'],
      include_dirs=[np.get_include()],
     )
