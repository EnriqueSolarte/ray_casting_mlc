# * load config resolvers
import geometry_perception_utils.config_utils
import os

RAY_CASTING_MLC_ROOT = os.path.dirname(os.path.abspath(__file__))
RAY_CASTING_MLC_CFG_DIR = os.path.join(RAY_CASTING_MLC_ROOT, 'config')

os.environ['RAY_CASTING_MLC_ROOT'] = RAY_CASTING_MLC_ROOT
os.environ['RAY_CASTING_MLC_CFG_DIR'] = RAY_CASTING_MLC_CFG_DIR
