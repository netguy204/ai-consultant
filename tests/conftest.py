import sys
import warnings

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore", message=".*no current event loop.*", category=DeprecationWarning)