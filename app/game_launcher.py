import subprocess
import sys
import os
from pathlib import Path

def run_game():
    """Dispara o jogo como subprocesso"""
    base_dir = Path(__file__).parent.parent
    game_script = base_dir / 'game' / 'game.py'
    python_exec = sys.executable  
    subprocess.Popen([python_exec, str(game_script)])
