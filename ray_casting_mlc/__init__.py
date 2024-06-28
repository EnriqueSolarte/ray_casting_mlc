# * load config resolvers
import geometry_perception_utils.config_utils
import os

MLC_PP_ROOT = os.path.dirname(os.path.abspath(__file__))
MLC_PP_CFG_DIR = os.path.join(MLC_PP_ROOT, 'config')

os.environ['MLC_PP_ROOT'] = MLC_PP_ROOT
os.environ['MLC_PP_CFG_DIR'] = MLC_PP_CFG_DIR
