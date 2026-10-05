from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from round4_attention.impl.cli import main
if __name__=='__main__':main('round9_low_precision','fp8_mlp_cur')
