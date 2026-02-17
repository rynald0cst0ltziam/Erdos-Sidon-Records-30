
import sys
import os
import platform
import scipy
import numpy
import datetime

def get_env_info():
    return {
        "python_version": sys.version,
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor(),
        "scipy_version": scipy.__version__,
        "numpy_version": numpy.__version__,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "command_line": " ".join(sys.argv)
    }
