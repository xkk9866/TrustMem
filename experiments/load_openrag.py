import marshal
import sys
import types
from pathlib import Path

path = Path("__pycache__/run_openrag.cpython-313.pyc")
data = path.read_bytes()
# Python 3.7+ header is 16 bytes.
code = marshal.loads(data[16:])
module = types.ModuleType("run_openrag")
module.__file__ = str(path)
sys.modules["run_openrag"] = module
exec(code, module.__dict__)
print([name for name in dir(module) if not name.startswith("_")])
