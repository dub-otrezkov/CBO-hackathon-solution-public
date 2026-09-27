N = 5
S, V, B, C, W = range(N)


def _matmul(A, B_):
    return [[sum(A[i][k] * B_[k][j] for k in range(N)) for j in range(N)] for i in range(N)]


def _transpose(A):
    return [[A[j][i] for j in range(N)] for i in range(N)]


class LongitudinalFilter:
    def __init__(self, q_accel: float = 0.05, q_bias: float = 0.0, q_gain: float = 0.0, q_scale: float = 0.0,
                 v0_var: float = 25.0, b0_var: float = 0.04, c0_var: float = 0.01, w0_var: float = 0.0004,
                 bias_limit: float = 1.0, gain_limits=(0.7, 1.4), scale_limits=(0.95, 1.05)):
        self.q_accel = q_accel
        self.q_bias = q_bias
        self.q_gain = q_gain
        self.q_scale = q_scale
        self.bias_limit = bias_limit
        self.gain_limits = gain_limits
        self.scale_limits = scale_limits
        self.x = [0.0, 0.0, 0.0, 1.0, 1.0]
        self.P = [[0.0] * N for _ in range(N)]
        self.P[V][V], self.P[B][B], self.P[C][C], self.P[W][W] = v0_var, b0_var, c0_var, w0_var

    @property
    def s(self) -> float:
        return self.x[S]

    @property
    def v(self) -> float:
        return self.x[V]

    @property
    def bias(self) -> float:
        return self.x[B]

    @property
    def gain(self) -> float:
        return self.x[C]

    @property
    def scale(self) -> float:
        return self.x[W]

    def accel(self, a_table: float, a_grade: float) -> float:
        return self.x[C] * a_table + a_grade + self.x[B]

    def predict(self, dt: float, a_table: float, a_grade: float = 0.0, hold: bool = False) -> None:
        if dt <= 0.0:
            return
        s, v, b, c, w = self.x
        P = self.P
        if hold:
            self.x = [s, 0.0, b, c, w]
            P[V][V] += 1e-4 * dt
            P[B][B] += self.q_bias * dt
            P[C][C] += self.q_gain * dt
            P[W][W] += self.q_scale * dt
            return
        a = c * a_table + a_grade + b
        v_new = v + a * dt
        if v_new < 0.0:
            t_stop = v / -a if a < 0.0 else dt
            s_new = s + v * t_stop + 0.5 * a * t_stop * t_stop
            v_new = 0.0
        else:
            s_new = s + v * dt + 0.5 * a * dt * dt
        self.x = [s_new, v_new, b, c, w]
        h = 0.5 * dt * dt
        F = [[1.0, dt, h, h * a_table, 0.0],
             [0.0, 1.0, dt, dt * a_table, 0.0],
             [0.0, 0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 1.0]]
        newP = _matmul(_matmul(F, P), _transpose(F))
        qa = self.q_accel * dt
        newP[S][S] += qa * dt * dt / 3.0
        newP[S][V] += qa * dt / 2.0
        newP[V][S] += qa * dt / 2.0
        newP[V][V] += qa
        newP[B][B] += self.q_bias * dt
        newP[C][C] += self.q_gain * dt
        newP[W][W] += self.q_scale * dt
        self.P = newP

    def _update(self, y: float, H: list, r: float) -> None:
        P = self.P
        PH = [sum(P[i][k] * H[k] for k in range(N)) for i in range(N)]
        S_ = sum(H[i] * PH[i] for i in range(N)) + r
        if S_ <= 0.0:
            return
        K = [PH[i] / S_ for i in range(N)]
        self.x = [self.x[i] + K[i] * y for i in range(N)]
        HP = [sum(H[k] * P[k][j] for k in range(N)) for j in range(N)]
        self.P = [[P[i][j] - K[i] * HP[j] for j in range(N)] for i in range(N)]
        self.x[V] = max(0.0, self.x[V])
        self.x[B] = max(-self.bias_limit, min(self.bias_limit, self.x[B]))
        self.x[C] = max(self.gain_limits[0], min(self.gain_limits[1], self.x[C]))
        self.x[W] = max(self.scale_limits[0], min(self.scale_limits[1], self.x[W]))

    def _wheel_h(self):
        return [0.0, self.x[W], 0.0, 0.0, self.x[V]]

    def innovation(self, z: float) -> tuple[float, float]:
        H = self._wheel_h()
        var = sum(H[i] * sum(self.P[i][k] * H[k] for k in range(N)) for i in range(N))
        return z - self.x[W] * self.x[V], var

    def update_wheel(self, z: float, r: float) -> None:
        self._update(z - self.x[W] * self.x[V], self._wheel_h(), r)

    def update_speed(self, z: float, r: float) -> None:
        self._update(z - self.x[V], [0.0, 1.0, 0.0, 0.0, 0.0], r)

    def update_position(self, z: float, r: float) -> None:
        self._update(z - self.x[S], [1.0, 0.0, 0.0, 0.0, 0.0], r)
