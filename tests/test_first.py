import math
from pathlib import Path
import os

import numpy as np
import pytest
import WH_RMM_forecasting.utils.WHtools as whtools 
import WH_RMM_forecasting.utils.ProcessForecasts as ProFo 
import WH_RMM_forecasting.utils.ProcessOBS as ProObs

# FIXME: Define paths as fixture at a central point
