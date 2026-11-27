"""Create an isolated environment for open-weight text-to-image generation.

The existing encoder environments are not modified.
"""
import os
import subprocess
from pathlib import Path

UV = os.path.expanduser('~/.local/bin/uv')
ENV = os.path.expanduser('~/recvis/.venv-gen')
PY = ENV + '/bin/python'
commands = [
    [UV, 'venv', '--python', '3.11', ENV],
    [UV, 'pip', 'install', '--python', PY, 'torch==2.4.1+cu121',
     '--index-url', 'https://download.pytorch.org/whl/cu121'],
    [UV, 'pip', 'install', '--python', PY, 'diffusers==0.31.0', 'transformers==4.46.3',
     'tokenizers==0.20.3', 'accelerate==1.1.1', 'safetensors==0.4.5',
     'huggingface_hub==0.26.5', 'numpy==1.26.4', 'pillow==11.3.0'],
]
for cmd in commands:
    print(cmd, flush=True)
    subprocess.run(cmd, check=True)
freeze = subprocess.run([UV, 'pip', 'freeze', '--python', PY], check=True,
                        capture_output=True, text=True).stdout
Path('runs/requirements-gen-freeze.txt').write_text(freeze)
print(freeze)
print('GEN_ENV_READY', flush=True)
