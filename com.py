DEBUG= False

FREQ_MINIMUM= 20#hz
FREQ_MAXIMUM= 2**13#hz

from math import floor
fftsize_calc= lambda rate: floor(rate*FREQ_MAXIMUM/FREQ_MINIMUM)

from dataclasses import dataclass as dcls