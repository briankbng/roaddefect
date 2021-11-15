from pathlib import Path

# This is your project root path.
ROOT_DIR = Path(__file__)

# This is the project models path.
MODELS_DIR = Path.joinpath(ROOT_DIR, 'models')

# This is the project data path.
DATA_DIR = Path.joinpath(ROOT_DIR, 'data')

# The path name where the definitions reside or the path name of the root package
SRC_ROOT = Path(__file__).parent



#
# Use below if you want to import definitions at the module package path level.
#
# import os.path
# import sys

# # Insert the SRC root path to the python system in order to get the definitions 
# # pacakge.
# current_dir = os.path.dirname(__file__)
# parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
# sys.path.insert(0, parent_dir)
# from definitions import SRC_VIS_ROOT
