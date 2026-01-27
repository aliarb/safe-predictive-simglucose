"""
Event-triggered insulin injection layer for NMPC/PID maintenance control.

This module adds a "basal-bolus like" mechanism on top of an existing controller:
- Keep the base controller (e.g. NMPCController in supervisor mode) for maintenance.
- Use a short-term + long-term Gaussian Process (GP) BG forecaster to *predict* near-future BG.
- Trigger short, time-limited bolus-rate pulses for predicted hyperglycemia (careful when uncertain).
- Trigger insulin suspension/reduction for predicted hypoglycemia (safety-first).

Notes on units:
- The simulator uses Action(basal, bolus) where *both* are insulin rates in U/min.
- The pump discretizes and clamps both basal and bolus separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Sequence, Tuple

import numpy as np

from .base import Action, Controller


def _rbf_kernel(x1: np.ndarray, x2: np.ndarray, *, lengthscale: float, variance: float) -> np.ndarray:
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.ndim == 1:
        x1 = x1[:, None]
    if x2.ndim == 1:
        x2 = x2[:, None]
    d2 = np.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
    return float(variance) * np.exp(-0.5 * d2 / float(lengthscale) ** 2)


@dataclass
class GPConfig:
    lengthscale: float
    variance: float
    noise: float


class _LiteGP:
    """Tiny RBF GP regressor for small datasets (fixed hyperparameters)."""

    def __init__(self, cfg: GPConfig):
        self.cfg = cfg
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self._y_mean: float = 140.0
        self._L: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim == 1:
            X = X[:, None]
        n = X.shape[0]
        if n < 2:
            self.X, self.y, self._L, self._alpha = X, y, None, None
            return

        # Constant-mean GP: fit on demeaned targets, add mean back at prediction time.
        # This avoids pathological behavior where the GP reverts to 0 mg/dL far from data.
        ym = float(np.nanmean(y))
        if not np.isfinite(ym):
            ym = 140.0
        self._y_mean = ym
        y0 = y - ym

        K = _rbf_kernel(X, X, lengthscale=self.cfg.lengthscale, variance=self.cfg.variance)
        K = K + (float(self.cfg.noise) ** 2) * np.eye(n)
        L = np.linalg.cholesky(K + 1e-10 * np.eye(n))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y0))
        self.X, self.y, self._L, self._alpha = X, y, L, alpha

    def predict(self, Xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xs = np.asarray(Xs, dtype=float)
        if Xs.ndim == 1:
            Xs = Xs[:, None]
        m = Xs.shape[0]
        if self.X is None or self.y is None or self.X.shape[0] < 2 or self._L is None or self._alpha is None:
            mean = np.full(m, 140.0, dtype=float)
            std = np.full(m, 50.0, dtype=float)
            return mean, std

        Kxs = _rbf_kernel(self.X, Xs, lengthscale=self.cfg.lengthscale, variance=self.cfg.variance)  # (n, m)
        mean = (Kxs.T @ self._alpha) + float(self._y_mean)
        v = np.linalg.solve(self._L, Kxs)
        Kss = _rbf_kernel(Xs, Xs, lengthscale=self.cfg.lengthscale, variance=self.cfg.variance)
        var = np.clip(np.diag(Kss) - np.sum(v * v, axis=0), 1e-8, np.inf)
        std = np.sqrt(var)
        return mean, std


class DualMemoryBGForecaster:
    """
    Two GP models:
    - Short-term: recent BG vs minutes-from-now.
    - Long-term: BG vs time-of-day embedding (sin/cos).
    Combined by inverse-variance weighting.
    """

    def __init__(
        self,
        *,
        short_window_minutes: float = 180.0,
        long_max_points: int = 2000,
        short_gp: GPConfig = GPConfig(lengthscale=25.0, variance=400.0, noise=5.0),
        long_gp: GPConfig = GPConfig(lengthscale=0.4, variance=300.0, noise=7.0),
    ):
        self.short_window_minutes = float(short_window_minutes)
        self.long_max_points = int(long_max_points)
        self._short = _LiteGP(short_gp)
        self._long = _LiteGP(long_gp)

        self._short_t: list[float] = []
        self._short_bg: list[float] = []
        self._long_feat: list[np.ndarray] = []
        self._long_bg: list[float] = []

    @staticmethod
    def _time_to_minutes(t: datetime) -> float:
        return float(t.timestamp() / 60.0)

    @staticmethod
    def _tod_features(t: datetime) -> np.ndarray:
        tod_min = t.hour * 60.0 + t.minute + t.second / 60.0
        ang = 2.0 * np.pi * (tod_min / 1440.0)
        return np.array([np.sin(ang), np.cos(ang)], dtype=float)

    def update(self, t: datetime, bg: float) -> None:
        bg_f = float(bg)
        if not np.isfinite(bg_f):
            return
        tm = self._time_to_minutes(t)

        self._short_t.append(tm)
        self._short_bg.append(bg_f)
        self._long_feat.append(self._tod_features(t))
        self._long_bg.append(bg_f)

        cutoff = tm - self.short_window_minutes
        while len(self._short_t) >= 2 and self._short_t[0] < cutoff:
            self._short_t.pop(0)
            self._short_bg.pop(0)

        t_arr = np.asarray(self._short_t, dtype=float)
        y_arr = np.asarray(self._short_bg, dtype=float)
        t0 = t_arr[-1]
        X_short = (t_arr - t0).reshape(-1, 1)
        self._short.fit(X_short, y_arr)

        if len(self._long_feat) > self.long_max_points:
            extra = len(self._long_feat) - self.long_max_points
            self._long_feat = self._long_feat[extra:]
            self._long_bg = self._long_bg[extra:]

        X_long = np.vstack(self._long_feat)
        y_long = np.asarray(self._long_bg, dtype=float)
        self._long.fit(X_long, y_long)

    def predict(self, t_now: datetime, future_minutes: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        future_minutes = np.asarray(list(future_minutes), dtype=float)
        Xs_short = future_minutes.reshape(-1, 1)
        m_s, s_s = self._short.predict(Xs_short)

        tf_dt: list[datetime] = [t_now + timedelta(minutes=float(m)) for m in future_minutes]
        Xs_long = np.vstack([self._tod_features(ti) for ti in tf_dt])
        m_l, s_l = self._long.predict(Xs_long)

        v_s = np.maximum(s_s ** 2, 1e-6)
        v_l = np.maximum(s_l ** 2, 1e-6)
        prec = (1.0 / v_s) + (1.0 / v_l)
        v = 1.0 / np.maximum(prec, 1e-9)
        m = v * (m_s / v_s + m_l / v_l)
        s = np.sqrt(v)
        return m, s


class EventTriggeredNMPCController(Controller):
    """
    Wrap an existing controller and apply event-triggered bolus-rate pulses / suspension
    based on GP forecasts.
    """

    def __init__(
        self,
        base: Controller,
        *,
        target_bg: float = 140.0,
        # Current-BG / rise anticipation knobs
        bg_trigger: float = 155.0,
        slope_trigger: float = 0.8,   # mg/dL/min (predicted positive slope)
        accel_trigger: float = 0.08,  # mg/dL/min^2 (predicted positive acceleration)
        # Observed (measured) BG derivative triggers (more reliable around meals than GP mean)
        obs_slope_trigger: float = 0.6,   # mg/dL/min
        obs_accel_trigger: float = 0.04,  # mg/dL/min^2
        # Meal-aware pulse shaping / limits (to push pulses earlier and avoid late extra pulses)
        postmeal_window_minutes: float = 180.0,
        max_hyper_pulses_per_meal: int = 2,
        meal_detect_threshold_g_per_min: float = 0.1,
        postmeal_boost_factor: float = 1.8,
        meal_new_gap_minutes: float = 120.0,
        hypo_threshold: float = 70.0,
        hyper_threshold: float = 180.0,
        prediction_horizon_minutes: float = 30.0,
        pulse_max_u_per_min: float = 2.0,
        pulse_minutes: float = 5.0,
        cooldown_minutes: float = 30.0,
        suspend_minutes: float = 15.0,
        uncertainty_k: float = 1.0,
        verbose: bool = True,
        forecaster: Optional[DualMemoryBGForecaster] = None,
    ):
        super().__init__(init_state=np.zeros(1))
        self.base = base
        self.target_bg = float(target_bg)
        self.bg_trigger = float(bg_trigger)
        self.slope_trigger = float(slope_trigger)
        self.accel_trigger = float(accel_trigger)
        self.obs_slope_trigger = float(obs_slope_trigger)
        self.obs_accel_trigger = float(obs_accel_trigger)
        self.postmeal_window_minutes = float(postmeal_window_minutes)
        self.max_hyper_pulses_per_meal = int(max_hyper_pulses_per_meal)
        self.meal_detect_threshold_g_per_min = float(meal_detect_threshold_g_per_min)
        self.postmeal_boost_factor = float(postmeal_boost_factor)
        self.meal_new_gap_minutes = float(meal_new_gap_minutes)
        self.hypo_threshold = float(hypo_threshold)
        self.hyper_threshold = float(hyper_threshold)
        self.prediction_horizon_minutes = float(prediction_horizon_minutes)
        self.pulse_max_u_per_min = float(pulse_max_u_per_min)
        self.pulse_minutes = float(pulse_minutes)
        self.cooldown_minutes = float(cooldown_minutes)
        self.suspend_minutes = float(suspend_minutes)
        self.uncertainty_k = float(uncertainty_k)
        self.verbose = bool(verbose)
        self.forecaster = forecaster if forecaster is not None else DualMemoryBGForecaster()

        self._cooldown_steps = 0
        self._pulse_steps = 0
        self._suspend_steps = 0
        self._active_pulse_u_per_min = 0.0

        # Debug trace (joined into results DF)
        self._dbg_time: list[datetime] = []
        self._dbg_bg_now: list[float] = []
        self._dbg_min_lower: list[float] = []
        self._dbg_max_upper: list[float] = []
        self._dbg_mu_max: list[float] = []
        self._dbg_sigma_mean: list[float] = []
        # Predicted derivatives from GP mean (mu):
        # d1: slope (mg/dL/min), d2: acceleration (mg/dL/min^2), d3: jerk (mg/dL/min^3)
        self._dbg_d1_mu_max: list[float] = []
        self._dbg_d1_mu_min: list[float] = []
        self._dbg_d2_mu_max: list[float] = []
        self._dbg_d2_mu_min: list[float] = []
        self._dbg_d3_mu_max: list[float] = []
        self._dbg_d3_mu_min: list[float] = []
        # Observed derivatives from recent BG measurements
        self._dbg_d1_obs: list[float] = []
        self._dbg_d2_obs: list[float] = []
        self._dbg_d3_obs: list[float] = []
        self._dbg_pulse_u_per_min: list[float] = []
        self._dbg_event: list[str] = []
        self._dbg_meal_rate: list[float] = []
        self._dbg_in_postmeal: list[int] = []
        self._dbg_hyper_pulses_since_meal: list[int] = []

        # Short observed history for derivative estimation (time, BG)
        self._obs_time: list[datetime] = []
        self._obs_bg: list[float] = []

        # Meal-aware tracking (based on info['meal'] in g/min)
        self._last_meal_time: Optional[datetime] = None
        self._hyper_pulses_since_meal: int = 0

    def reset(self):
        self._cooldown_steps = 0
        self._pulse_steps = 0
        self._suspend_steps = 0
        self._active_pulse_u_per_min = 0.0
        self._dbg_time.clear()
        self._dbg_bg_now.clear()
        self._dbg_min_lower.clear()
        self._dbg_max_upper.clear()
        self._dbg_mu_max.clear()
        self._dbg_sigma_mean.clear()
        self._dbg_d1_mu_max.clear()
        self._dbg_d1_mu_min.clear()
        self._dbg_d2_mu_max.clear()
        self._dbg_d2_mu_min.clear()
        self._dbg_d3_mu_max.clear()
        self._dbg_d3_mu_min.clear()
        self._dbg_d1_obs.clear()
        self._dbg_d2_obs.clear()
        self._dbg_d3_obs.clear()
        self._dbg_pulse_u_per_min.clear()
        self._dbg_event.clear()
        self._dbg_meal_rate.clear()
        self._dbg_in_postmeal.clear()
        self._dbg_hyper_pulses_since_meal.clear()
        self._obs_time.clear()
        self._obs_bg.clear()
        self._last_meal_time = None
        self._hyper_pulses_since_meal = 0
        try:
            self.base.reset()
        except Exception:
            pass

    def _steps(self, minutes: float, sample_time_min: float) -> int:
        return int(np.ceil(float(minutes) / max(float(sample_time_min), 1e-6)))

    def _append_debug(
        self,
        t_now,
        *,
        bg_now: float,
        min_lower: float,
        max_upper: float,
        mu_max: float,
        sigma_mean: float,
        d1_mu_max: float,
        d1_mu_min: float,
        d2_mu_max: float,
        d2_mu_min: float,
        d3_mu_max: float,
        d3_mu_min: float,
        d1_obs: float,
        d2_obs: float,
        d3_obs: float,
        pulse_u_per_min: float,
        event: str,
        meal_rate_g_per_min: float,
        in_postmeal: int,
        hyper_pulses_since_meal: int,
    ) -> None:
        if not isinstance(t_now, datetime):
            return
        self._dbg_time.append(t_now)
        self._dbg_bg_now.append(float(bg_now))
        self._dbg_min_lower.append(float(min_lower))
        self._dbg_max_upper.append(float(max_upper))
        self._dbg_mu_max.append(float(mu_max))
        self._dbg_sigma_mean.append(float(sigma_mean))
        self._dbg_d1_mu_max.append(float(d1_mu_max))
        self._dbg_d1_mu_min.append(float(d1_mu_min))
        self._dbg_d2_mu_max.append(float(d2_mu_max))
        self._dbg_d2_mu_min.append(float(d2_mu_min))
        self._dbg_d3_mu_max.append(float(d3_mu_max))
        self._dbg_d3_mu_min.append(float(d3_mu_min))
        self._dbg_d1_obs.append(float(d1_obs))
        self._dbg_d2_obs.append(float(d2_obs))
        self._dbg_d3_obs.append(float(d3_obs))
        self._dbg_pulse_u_per_min.append(float(pulse_u_per_min))
        self._dbg_event.append(str(event))
        self._dbg_meal_rate.append(float(meal_rate_g_per_min))
        self._dbg_in_postmeal.append(int(in_postmeal))
        self._dbg_hyper_pulses_since_meal.append(int(hyper_pulses_since_meal))

    def get_debug_df(self):
        """Return a pandas DataFrame indexed by Time with trigger diagnostics."""
        import pandas as pd

        if not self._dbg_time:
            return pd.DataFrame()
        df = pd.DataFrame(
            {
                "bg_now": self._dbg_bg_now,
                "min_lower": self._dbg_min_lower,
                "max_upper": self._dbg_max_upper,
                "mu_max": self._dbg_mu_max,
                "sigma_mean": self._dbg_sigma_mean,
                "d1_mu_max": self._dbg_d1_mu_max,
                "d1_mu_min": self._dbg_d1_mu_min,
                "d2_mu_max": self._dbg_d2_mu_max,
                "d2_mu_min": self._dbg_d2_mu_min,
                "d3_mu_max": self._dbg_d3_mu_max,
                "d3_mu_min": self._dbg_d3_mu_min,
                "d1_obs": self._dbg_d1_obs,
                "d2_obs": self._dbg_d2_obs,
                "d3_obs": self._dbg_d3_obs,
                "pulse_u_per_min": self._dbg_pulse_u_per_min,
                "event": self._dbg_event,
                "meal_rate": self._dbg_meal_rate,
                "in_postmeal": self._dbg_in_postmeal,
                "hyper_pulses_since_meal": self._dbg_hyper_pulses_since_meal,
            },
            index=pd.to_datetime(self._dbg_time),
        )
        df.index.name = "Time"
        return df

    def policy(self, observation, reward, done, **info):
        base_action = self.base.policy(observation, reward, done, **info)
        sample_time = float(info.get("sample_time", 5))

        t_now = info.get("time", None)
        bg_now = info.get("bg", None)
        if bg_now is None:
            bg_now = getattr(observation, "CGM", None)
        bg_now_f = float(bg_now) if bg_now is not None and np.isfinite(float(bg_now)) else float("nan")
        if isinstance(t_now, datetime) and bg_now is not None:
            self.forecaster.update(t_now, float(bg_now))

        # Meal detection (env provides `meal` in g/min; note: passed via **info)
        meal_rate = info.get("meal", 0.0)
        try:
            meal_rate_f = float(meal_rate)
        except Exception:
            meal_rate_f = 0.0
        if isinstance(t_now, datetime) and np.isfinite(meal_rate_f) and meal_rate_f > self.meal_detect_threshold_g_per_min:
            # New meal event: set meal *start* time and reset counter.
            if self._last_meal_time is None or (t_now - self._last_meal_time) > timedelta(minutes=float(self.meal_new_gap_minutes)):
                self._last_meal_time = t_now
                self._hyper_pulses_since_meal = 0

        in_postmeal = False
        if isinstance(t_now, datetime) and self._last_meal_time is not None:
            in_postmeal = (t_now - self._last_meal_time) <= timedelta(minutes=float(self.postmeal_window_minutes))

        # Update observed history and compute observed derivatives (at current time)
        d1_obs = float("nan")
        d2_obs = float("nan")
        d3_obs = float("nan")
        if isinstance(t_now, datetime) and np.isfinite(bg_now_f):
            self._obs_time.append(t_now)
            self._obs_bg.append(float(bg_now_f))
            # Keep last ~6 points (~30 minutes at 5-min control; more if env uses 3-min)
            if len(self._obs_time) > 8:
                self._obs_time = self._obs_time[-8:]
                self._obs_bg = self._obs_bg[-8:]
            if len(self._obs_time) >= 3:
                tmins = np.array([(ti - self._obs_time[0]).total_seconds() / 60.0 for ti in self._obs_time], dtype=float)
                bgarr = np.array(self._obs_bg, dtype=float)
                # Guard against duplicate timestamps
                if np.all(np.diff(tmins) > 0):
                    d1 = np.gradient(bgarr, tmins)
                    d2 = np.gradient(d1, tmins)
                    d3 = np.gradient(d2, tmins)
                    d1_obs = float(d1[-1])
                    d2_obs = float(d2[-1])
                    d3_obs = float(d3[-1])

        # timers
        if self._cooldown_steps > 0:
            self._cooldown_steps -= 1
        if self._pulse_steps > 0:
            self._pulse_steps -= 1
        if self._suspend_steps > 0:
            self._suspend_steps -= 1

        # default debug values
        min_lower = float("nan")
        max_upper = float("nan")
        mu_max = float("nan")
        sigma_mean = float("nan")
        d1_mu_max = float("nan")
        d1_mu_min = float("nan")
        d2_mu_max = float("nan")
        d2_mu_min = float("nan")
        d3_mu_max = float("nan")
        d3_mu_min = float("nan")
        # observed derivatives
        # (already computed above; keep as-is)

        if self._suspend_steps > 0:
            self._append_debug(
                t_now,
                bg_now=bg_now_f,
                min_lower=min_lower,
                max_upper=max_upper,
                mu_max=mu_max,
                sigma_mean=sigma_mean,
                d1_mu_max=d1_mu_max,
                d1_mu_min=d1_mu_min,
                d2_mu_max=d2_mu_max,
                d2_mu_min=d2_mu_min,
                d3_mu_max=d3_mu_max,
                d3_mu_min=d3_mu_min,
                d1_obs=d1_obs,
                d2_obs=d2_obs,
                d3_obs=d3_obs,
                pulse_u_per_min=0.0,
                event="hold_suspend",
                meal_rate_g_per_min=meal_rate_f,
                in_postmeal=int(in_postmeal),
                hyper_pulses_since_meal=int(self._hyper_pulses_since_meal),
            )
            return Action(basal=0.0, bolus=0.0)

        if self._pulse_steps > 0:
            bol = float(getattr(base_action, "bolus", 0.0))
            bas = float(getattr(base_action, "basal", 0.0))
            self._append_debug(
                t_now,
                bg_now=bg_now_f,
                min_lower=min_lower,
                max_upper=max_upper,
                mu_max=mu_max,
                sigma_mean=sigma_mean,
                d1_mu_max=d1_mu_max,
                d1_mu_min=d1_mu_min,
                d2_mu_max=d2_mu_max,
                d2_mu_min=d2_mu_min,
                d3_mu_max=d3_mu_max,
                d3_mu_min=d3_mu_min,
                d1_obs=d1_obs,
                d2_obs=d2_obs,
                d3_obs=d3_obs,
                pulse_u_per_min=float(self._active_pulse_u_per_min),
                event="hold_pulse",
                meal_rate_g_per_min=meal_rate_f,
                in_postmeal=int(in_postmeal),
                hyper_pulses_since_meal=int(self._hyper_pulses_since_meal),
            )
            return Action(basal=bas, bolus=bol + float(self._active_pulse_u_per_min))

        if not isinstance(t_now, datetime):
            self._append_debug(
                t_now,
                bg_now=bg_now_f,
                min_lower=min_lower,
                max_upper=max_upper,
                mu_max=mu_max,
                sigma_mean=sigma_mean,
                d1_mu_max=d1_mu_max,
                d1_mu_min=d1_mu_min,
                d2_mu_max=d2_mu_max,
                d2_mu_min=d2_mu_min,
                d3_mu_max=d3_mu_max,
                d3_mu_min=d3_mu_min,
                d1_obs=d1_obs,
                d2_obs=d2_obs,
                d3_obs=d3_obs,
                pulse_u_per_min=0.0,
                event="no_time",
                meal_rate_g_per_min=meal_rate_f,
                in_postmeal=int(in_postmeal),
                hyper_pulses_since_meal=int(self._hyper_pulses_since_meal),
            )
            return base_action

        # Forecast on a fixed grid (minutes ahead).
        # Include 0 min so we can calibrate the GP mean to the currently observed BG
        # (prevents biased envelopes that drift away from the real trace late in long runs).
        future_grid_all = np.arange(0.0, self.prediction_horizon_minutes + 1e-9, 5.0)
        mu_all, sig_all = self.forecaster.predict(t_now, future_grid_all)

        # Bias-correct the mean so mu(0) matches current BG (constant shift; preserves derivatives).
        mu_all = np.asarray(mu_all, dtype=float)
        sig_all = np.asarray(sig_all, dtype=float)
        if mu_all.size >= 1 and np.isfinite(mu_all[0]) and np.isfinite(bg_now_f):
            delta = float(bg_now_f - mu_all[0])
            # Keep correction bounded (avoid huge shifts if the GP is momentarily unstable).
            delta = float(np.clip(delta, -40.0, 40.0))
            mu_all = mu_all + delta

        # For triggering, use only the *future* portion (exclude 0 min).
        mu = mu_all[1:]
        sig = sig_all[1:]

        lower = mu - self.uncertainty_k * sig
        upper = mu + self.uncertainty_k * sig

        pred_low = float(np.min(lower))
        pred_high = float(np.max(upper))
        mu_max = float(np.max(mu))
        sigma_mean = float(np.mean(sig))
        min_lower = pred_low
        max_upper = pred_high

        # Derivatives of predicted mean BG (mu) over the forecast grid.
        # These are useful to detect "fast drop after pulse" risk.
        dt = 5.0  # minutes (matches our future grid step)
        if mu.size >= 3:
            d1 = np.gradient(mu, dt)
            d2 = np.gradient(d1, dt)
            d3 = np.gradient(d2, dt)
            d1_mu_max = float(np.max(d1))
            d1_mu_min = float(np.min(d1))
            d2_mu_max = float(np.max(d2))
            d2_mu_min = float(np.min(d2))
            d3_mu_max = float(np.max(d3))
            d3_mu_min = float(np.min(d3))

        if pred_low < self.hypo_threshold:
            self._suspend_steps = self._steps(self.suspend_minutes, sample_time)
            self._cooldown_steps = self._steps(self.cooldown_minutes, sample_time)
            if self.verbose:
                print(
                    f"[EventTriggered] SUSPEND for {self.suspend_minutes:.0f} min "
                    f"(min lower={pred_low:.1f} < {self.hypo_threshold:.1f})",
                    flush=True,
                )
            self._append_debug(
                t_now,
                bg_now=bg_now_f,
                min_lower=min_lower,
                max_upper=max_upper,
                mu_max=mu_max,
                sigma_mean=sigma_mean,
                d1_mu_max=d1_mu_max,
                d1_mu_min=d1_mu_min,
                d2_mu_max=d2_mu_max,
                d2_mu_min=d2_mu_min,
                d3_mu_max=d3_mu_max,
                d3_mu_min=d3_mu_min,
                d1_obs=d1_obs,
                d2_obs=d2_obs,
                d3_obs=d3_obs,
                pulse_u_per_min=0.0,
                event="trigger_hypo_suspend",
                meal_rate_g_per_min=meal_rate_f,
                in_postmeal=int(in_postmeal),
                hyper_pulses_since_meal=int(self._hyper_pulses_since_meal),
            )
            return Action(basal=0.0, bolus=0.0)

        # Hyper trigger:
        # - either classic "confident hyper" (upper bound crosses hyper_threshold),
        # - OR early-rise anticipation based on current BG + positive slope/acceleration.
        early_rise_pred = (
            np.isfinite(bg_now_f)
            and (bg_now_f >= self.bg_trigger)
            and (np.isfinite(d1_mu_max) and d1_mu_max >= self.slope_trigger)
        ) or (
            np.isfinite(d2_mu_max) and d2_mu_max >= self.accel_trigger and np.isfinite(d1_mu_max) and d1_mu_max > 0.0
        )
        # Observed early rise (push pulses earlier, e.g. right after meals)
        early_rise_obs = (
            np.isfinite(bg_now_f)
            and (bg_now_f >= (self.target_bg + 5.0))
            and np.isfinite(d1_obs)
            and (d1_obs >= self.obs_slope_trigger)
        ) or (
            np.isfinite(bg_now_f)
            and (bg_now_f >= (self.target_bg + 10.0))
            and np.isfinite(d2_obs)
            and (d2_obs >= self.obs_accel_trigger)
            and np.isfinite(d1_obs)
            and (d1_obs > 0.0)
        )

        early_rise = early_rise_pred or early_rise_obs
        confident_hyper = pred_high > self.hyper_threshold

        if self._cooldown_steps <= 0 and (confident_hyper or early_rise):
            # Hard limit: avoid >2 hyper pulses per meal window (prevents late 3rd pulse that can crash BG)
            if in_postmeal and self._hyper_pulses_since_meal >= self.max_hyper_pulses_per_meal:
                self._append_debug(
                    t_now,
                    bg_now=bg_now_f,
                    min_lower=min_lower,
                    max_upper=max_upper,
                    mu_max=mu_max,
                    sigma_mean=sigma_mean,
                    d1_mu_max=d1_mu_max,
                    d1_mu_min=d1_mu_min,
                    d2_mu_max=d2_mu_max,
                    d2_mu_min=d2_mu_min,
                    d3_mu_max=d3_mu_max,
                    d3_mu_min=d3_mu_min,
                    d1_obs=d1_obs,
                    d2_obs=d2_obs,
                    d3_obs=d3_obs,
                    pulse_u_per_min=0.0,
                    event="limit_max_hyper_pulses_per_meal",
                    meal_rate_g_per_min=meal_rate_f,
                    in_postmeal=int(in_postmeal),
                    hyper_pulses_since_meal=int(self._hyper_pulses_since_meal),
                )
                return base_action

            # Extra safety: if forecast already shows a strong downward curvature (fast drop),
            # avoid aggressive pulses that can push BG < 70 later.
            # Gate pulse if:
            # - lower bound is too close to hypo threshold, or
            # - strong negative acceleration/jerk is present (rapidly increasing downward trend).
            safety_margin = 10.0  # mg/dL margin above hypo_threshold
            accel_limit = -0.20   # mg/dL/min^2 (heuristic conservative)
            jerk_limit = -0.03    # mg/dL/min^3
            if (pred_low < (self.hypo_threshold + safety_margin)) or (np.isfinite(d2_mu_min) and d2_mu_min < accel_limit) or (np.isfinite(d3_mu_min) and d3_mu_min < jerk_limit):
                self._append_debug(
                    t_now,
                    bg_now=bg_now_f,
                    min_lower=min_lower,
                    max_upper=max_upper,
                    mu_max=mu_max,
                    sigma_mean=sigma_mean,
                    d1_mu_max=d1_mu_max,
                    d1_mu_min=d1_mu_min,
                    d2_mu_max=d2_mu_max,
                    d2_mu_min=d2_mu_min,
                    d3_mu_max=d3_mu_max,
                    d3_mu_min=d3_mu_min,
                    d1_obs=d1_obs,
                    d2_obs=d2_obs,
                    d3_obs=d3_obs,
                    pulse_u_per_min=0.0,
                    event="gate_hyper_pulse_due_to_drop_risk",
                    meal_rate_g_per_min=meal_rate_f,
                    in_postmeal=int(in_postmeal),
                    hyper_pulses_since_meal=int(self._hyper_pulses_since_meal),
                )
                return base_action

            # Pulse magnitude: include current BG and rise dynamics.
            bg_term = max(0.0, (bg_now_f - self.target_bg)) if np.isfinite(bg_now_f) else 0.0
            overshoot = max(0.0, mu_max - self.target_bg)
            rise_term_pred = max(0.0, d1_mu_max) * 10.0 + max(0.0, d2_mu_max) * 200.0  # scale to mg/dL-like
            rise_term_obs = max(0.0, d1_obs) * 12.0 + max(0.0, d2_obs) * 250.0
            rise_term = max(rise_term_pred, rise_term_obs)
            raw = 0.015 * overshoot + 0.006 * bg_term + 0.004 * rise_term
            # Downscale by uncertainty
            raw = raw / max(1.0, float(sigma_mean))
            # Post-meal boost: if we're in the post-meal window and seeing a clear rise,
            # allow a stronger earlier pulse (still clipped and still gated by min_lower).
            if in_postmeal and early_rise and (np.isfinite(bg_now_f) and bg_now_f >= (self.target_bg + 10.0)):
                raw = raw * float(self.postmeal_boost_factor)

            # If we're in post-meal and observed slope indicates a strong rise, allow "large" earlier pulses.
            if in_postmeal and early_rise_obs and np.isfinite(d1_obs) and np.isfinite(bg_now_f):
                # Build an additional candidate based directly on observed rise.
                # This is what shifts the big pulses earlier (18.5-19h) instead of waiting for max_upper>180.
                denom = max(1.0, float(sigma_mean) / 5.0)
                pulse_candidate = (0.25 * max(0.0, d1_obs) + 0.20 * max(0.0, (bg_now_f - self.target_bg) / 40.0)) / denom
                raw = max(raw, pulse_candidate)
            pulse = float(np.clip(raw, 0.0, self.pulse_max_u_per_min))
            if pulse > 0.0:
                self._pulse_steps = self._steps(self.pulse_minutes, sample_time)
                self._cooldown_steps = self._steps(self.cooldown_minutes, sample_time)
                self._active_pulse_u_per_min = float(pulse)
                if in_postmeal:
                    self._hyper_pulses_since_meal += 1
                if self.verbose:
                    print(
                        f"[EventTriggered] PULSE bolus={pulse:.2f} U/min for {self.pulse_minutes:.0f} min "
                        f"(reason={'confident_hyper' if confident_hyper else 'early_rise'}, avg σ={sigma_mean:.1f})",
                        flush=True,
                    )
                bas = float(getattr(base_action, "basal", 0.0))
                bol = float(getattr(base_action, "bolus", 0.0))
                self._append_debug(
                    t_now,
                    bg_now=bg_now_f,
                    min_lower=min_lower,
                    max_upper=max_upper,
                    mu_max=mu_max,
                    sigma_mean=sigma_mean,
                    d1_mu_max=d1_mu_max,
                    d1_mu_min=d1_mu_min,
                    d2_mu_max=d2_mu_max,
                    d2_mu_min=d2_mu_min,
                    d3_mu_max=d3_mu_max,
                    d3_mu_min=d3_mu_min,
                    d1_obs=d1_obs,
                    d2_obs=d2_obs,
                    d3_obs=d3_obs,
                    pulse_u_per_min=float(pulse),
                    event="trigger_hyper_pulse",
                    meal_rate_g_per_min=meal_rate_f,
                    in_postmeal=int(in_postmeal),
                    hyper_pulses_since_meal=int(self._hyper_pulses_since_meal),
                )
                return Action(basal=bas, bolus=bol + pulse)

        self._append_debug(
            t_now,
            bg_now=bg_now_f,
            min_lower=min_lower,
            max_upper=max_upper,
            mu_max=mu_max,
            sigma_mean=sigma_mean,
            d1_mu_max=d1_mu_max,
            d1_mu_min=d1_mu_min,
            d2_mu_max=d2_mu_max,
            d2_mu_min=d2_mu_min,
            d3_mu_max=d3_mu_max,
            d3_mu_min=d3_mu_min,
            d1_obs=d1_obs,
            d2_obs=d2_obs,
            d3_obs=d3_obs,
            pulse_u_per_min=0.0,
            event="none",
            meal_rate_g_per_min=meal_rate_f,
            in_postmeal=int(in_postmeal),
            hyper_pulses_since_meal=int(self._hyper_pulses_since_meal),
        )
        return base_action

