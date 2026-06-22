"""Launch VA_Server in websocket mode without torchrun/dist init."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wan_va.wan_va_server import VA_Server, run_async_server_mode
from wan_va.configs import VA_CONFIGS
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config-name', type=str, default='libero')
parser.add_argument('--port', type=int, default=29056)
parser.add_argument('--save-root', type=str, default='visualization/')
args = parser.parse_args()

config = VA_CONFIGS[args.config_name]
config.rank = 0
config.local_rank = 0
config.world_size = 1
config.save_root = args.save_root

print('Loading model...')
model = VA_Server(config)
print(f'Server starting on port {args.port}...')
run_async_server_mode(model, 0, config.host, args.port)
