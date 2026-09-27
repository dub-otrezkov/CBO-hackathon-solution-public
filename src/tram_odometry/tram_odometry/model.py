from bisect import bisect_right
from collections import deque

DELAY_S = 0.2
GRADE_GAIN = 0.7
GRAVITY = 9.81
SPEED_BP = (0.0, 1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0)
MIN_NOTCH = -15
MAX_NOTCH = 15
TABLE = (
    (-0.9140, -0.2740, -0.2850, -1.3980, -1.4090, -1.4090, -1.4090, -1.4090),
    (-1.0580, -1.1320, -1.1180, -1.5900, -1.5990, -1.5990, -1.5990, -1.5990),
    (-1.0430, -1.1530, -1.2120, -1.3600, -1.3210, -1.3210, -1.3210, -1.3210),
    (-0.9950, -1.1100, -1.1140, -1.3450, -1.3170, -1.2740, -1.2740, -1.2740),
    (-0.9480, -1.0580, -1.1240, -1.2740, -1.2540, -1.2100, -1.2100, -1.2100),
    (-0.8860, -0.9900, -1.0350, -1.1480, -1.1690, -1.1360, -1.1360, -1.1360),
    (-0.8070, -0.9300, -0.9400, -1.0060, -1.0460, -1.0040, -1.0040, -1.0040),
    (-0.7330, -0.8660, -0.9060, -0.7680, -0.3590, -0.3120, -0.1160, -0.1160),
    (-0.6570, -0.7990, -0.8320, -0.8570, -0.8590, -0.8340, -0.8280, -0.8280),
    (-0.5970, -0.7200, -0.7390, -0.7680, -0.7820, -0.7710, -0.7680, -0.7680),
    (-0.5240, -0.6600, -0.6770, -0.6970, -0.7200, -0.7090, -0.6830, -0.6830),
    (-0.4580, -0.5770, -0.6000, -0.6410, -0.6650, -0.6460, -0.6020, -0.6020),
    (-0.4050, -0.5160, -0.5290, -0.5700, -0.5880, -0.5540, -0.4850, -0.4850),
    (-0.3720, -0.4370, -0.4290, -0.4140, -0.4080, -0.4050, -0.3460, -0.3460),
    (-0.3080, -0.2840, -0.2480, -0.2170, -0.2070, -0.2120, -0.2030, -0.2030),
    (-0.0580, -0.0780, -0.0710, -0.0320, -0.0290, -0.0220, -0.0270, -0.0270),
    (0.0530, 0.0610, 0.0290, 0.0880, 0.0860, 0.0750, 0.0700, 0.0700),
    (0.2140, 0.1960, 0.1670, 0.2070, 0.1690, 0.1470, 0.1340, 0.1340),
    (0.3700, 0.3670, 0.3020, 0.2930, 0.2270, 0.1920, 0.1750, 0.1750),
    (0.4470, 0.4790, 0.4370, 0.3920, 0.2270, 0.2050, 0.1900, 0.1900),
    (0.5630, 0.6100, 0.5600, 0.4600, 0.3570, 0.2670, 0.2340, 0.2340),
    (0.6550, 0.7270, 0.6490, 0.5630, 0.4510, 0.3130, 0.2560, 0.2560),
    (0.7300, 0.7980, 0.7580, 0.7010, 0.5140, 0.3780, 0.3150, 0.3150),
    (0.7970, 0.8900, 0.8630, 0.7930, 0.5830, 0.4210, 0.3480, 0.3480),
    (0.8400, 0.9330, 0.9200, 0.8550, 0.6300, 0.4640, 0.3830, 0.3830),
    (0.9240, 0.9240, 0.9360, 0.8780, 0.7020, 0.5120, 0.4230, 0.4230),
    (0.8920, 0.8920, 0.9440, 0.8880, 0.7350, 0.5590, 0.4790, 0.4790),
    (0.9210, 0.9210, 0.9550, 0.8840, 0.7610, 0.5800, 0.5030, 0.5030),
    (0.9140, 0.9140, 0.9810, 0.8820, 0.7780, 0.5990, 0.5470, 0.5470),
    (1.0870, 1.0870, 1.0270, 0.8770, 0.7730, 0.6480, 0.5810, 0.5810),
    (0.9140, 0.9140, 0.9140, 0.8580, 0.7220, 0.6350, 0.5740, 0.5740))


M8_FROM_ABOVE = (-0.733, -0.866, -0.906, -0.768, -0.359, -0.312, -0.116, -0.116)
M8_FROM_BELOW = (-0.239, -0.239, -0.107, -0.089, -0.112, -0.110, -0.110, -0.110)


def _interp_row(row, v: float) -> float:
    v = max(0.0, v)
    if v >= SPEED_BP[-1]:
        return row[-1]
    i = bisect_right(SPEED_BP, v) - 1
    f = (v - SPEED_BP[i]) / (SPEED_BP[i + 1] - SPEED_BP[i])
    return row[i] * (1.0 - f) + row[i + 1] * f


def table_accel(notch: int, v: float) -> float:
    row = TABLE[max(MIN_NOTCH, min(MAX_NOTCH, int(notch))) - MIN_NOTCH]
    v = max(0.0, v)
    if v >= SPEED_BP[-1]:
        return row[-1]
    i = bisect_right(SPEED_BP, v) - 1
    f = (v - SPEED_BP[i]) / (SPEED_BP[i + 1] - SPEED_BP[i])
    return row[i] * (1.0 - f) + row[i + 1] * f


class TractionModel:

    def __init__(self, delay_s: float = DELAY_S, history_s: float = 5.0, m8_hysteresis: bool = True):
        self.delay_s = delay_s
        self.history_s = history_s
        self.m8_hysteresis = m8_hysteresis
        self._notches = deque()
        self.notch = 0
        self._from_below = False

    def on_controller(self, stamp: float, notch: int) -> None:
        notch = max(MIN_NOTCH, min(MAX_NOTCH, int(notch)))
        if notch == -8 and self.notch != -8:
            self._from_below = self.notch <= -9
        if self._notches and stamp < self._notches[-1][0]:
            stamp = self._notches[-1][0]
        self._notches.append((stamp, notch, notch == -8 and self._from_below))
        while len(self._notches) > 2 and self._notches[1][0] < stamp - self.history_s:
            self._notches.popleft()
        self.notch = notch

    def _entry_at(self, stamp: float):
        target = stamp - self.delay_s
        effective = self._notches[0] if self._notches else (0.0, 0, False)
        for e in self._notches:
            if e[0] > target:
                break
            effective = e
        return effective

    def notch_at(self, stamp: float) -> int:
        return self._entry_at(stamp)[1]

    def table(self, stamp: float, v: float) -> float:
        _t, notch, from_below = self._entry_at(stamp)
        if notch == -8 and self.m8_hysteresis:
            return _interp_row(M8_FROM_BELOW if from_below else M8_FROM_ABOVE, v)
        return table_accel(notch, v)

    @staticmethod
    def grade_accel(grade: float) -> float:
        return -GRADE_GAIN * GRAVITY * grade

    def accel(self, stamp: float, v: float, grade: float = 0.0) -> float:
        return self.table(stamp, v) + self.grade_accel(grade)
