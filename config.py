import os
os.environ['NCCL_P2P_DISABLE'] = '1'

project_root = os.path.dirname(os.path.realpath(__file__))