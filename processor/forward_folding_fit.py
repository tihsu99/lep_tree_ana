from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

ONE_SIGMA_DELTA_NEG2_LOG_LIKELIHOOD = 1.0
ONE_SIGMA_CONFIDENCE_LEVEL = float(chi2.cdf(ONE_SIGMA_DELTA_NEG2_LOG_LIKELIHOOD, df=1))


@dataclass(frozen=True)
class NuisanceParameterSpec:
    name: str
    initial_value: float = 1.0
    bounds: tuple[float, float] = (0.0, 2.0)
    fit: bool = False
    constraint_sigma: float | None = None


@dataclass(frozen=True)
class FitParameterSnapshot:
    pois: OrderedDict = field(default_factory=OrderedDict)
    nuisance_parameters: OrderedDict = field(default_factory=OrderedDict)

    def to_dict(self):
        return {
            "pois": dict(self.pois),
            "nuisance_parameters": dict(self.nuisance_parameters),
        }


@dataclass(frozen=True)
class LikelihoodIntervalResult:
    label: str
    threshold_delta_neg2_log_likelihood: float
    confidence_level: float | None = None
    tail_probability_definition: str = "custom_threshold"
    interval_lower: float | None = None
    interval_upper: float | None = None

    def to_dict(self):
        return {
            "label": self.label,
            "threshold_delta_neg2_log_likelihood": float(
                self.threshold_delta_neg2_log_likelihood
            ),
            "confidence_level": self.confidence_level,
            "tail_probability_definition": self.tail_probability_definition,
            "interval_lower": self.interval_lower,
            "interval_upper": self.interval_upper,
        }


@dataclass(frozen=True)
class LikelihoodScanResult:
    poi_values: tuple[float, ...] = ()
    neg2_log_likelihood_values: tuple[float, ...] = ()
    delta_neg2_log_likelihood_values: tuple[float, ...] = ()
    one_sigma_interval: LikelihoodIntervalResult | None = None
    intervals: tuple[LikelihoodIntervalResult, ...] = ()

    @property
    def threshold_delta_neg2_log_likelihood(self):
        return None if self.one_sigma_interval is None else self.one_sigma_interval.threshold_delta_neg2_log_likelihood

    @property
    def confidence_level(self):
        return None if self.one_sigma_interval is None else self.one_sigma_interval.confidence_level

    @property
    def tail_probability_definition(self):
        return None if self.one_sigma_interval is None else self.one_sigma_interval.tail_probability_definition

    @property
    def interval_lower(self):
        return None if self.one_sigma_interval is None else self.one_sigma_interval.interval_lower

    @property
    def interval_upper(self):
        return None if self.one_sigma_interval is None else self.one_sigma_interval.interval_upper

    def to_dict(self):
        return {
            "poi_values": list(self.poi_values),
            "neg2_log_likelihood_values": list(self.neg2_log_likelihood_values),
            "delta_neg2_log_likelihood_values": list(
                self.delta_neg2_log_likelihood_values
            ),
            "one_sigma_interval": (
                None if self.one_sigma_interval is None else self.one_sigma_interval.to_dict()
            ),
            "intervals": [interval.to_dict() for interval in self.intervals],
            "threshold_delta_neg2_log_likelihood": self.threshold_delta_neg2_log_likelihood,
            "confidence_level": self.confidence_level,
            "tail_probability_definition": self.tail_probability_definition,
            "interval_lower": self.interval_lower,
            "interval_upper": self.interval_upper,
        }


@dataclass(frozen=True)
class SinglePOIFitResult:
    prefit: FitParameterSnapshot
    postfit: FitParameterSnapshot
    poi_uncertainty_up: float
    poi_uncertainty_down: float
    neg2_log_likelihood: float
    optimizer_result: object
    likelihood_scan: LikelihoodScanResult | None = None

    @property
    def chi2(self):
        return self.neg2_log_likelihood

    @property
    def poi_uncertainty(self):
        return {
            "up": self.poi_uncertainty_up,
            "down": self.poi_uncertainty_down,
        }


class SinglePOIFitter:
    """
    Profile one parameter of interest with a binned Poisson likelihood.

    The detector/model-specific prediction is intentionally supplied as a
    callback so this class stays as a small statistical fitting helper.
    """

    def __init__(
        self,
        poi_name,
        nominal_poi_value,
        poi_bounds,
        nuisance_parameter_specs,
        data_values,
        data_errors,
        build_expected_values,
        uncertainty_method="Likelihood_Scan",
        likelihood_scan_points=101,
        likelihood_scan_thresholds=None,
        likelihood_scan_confidence_levels=None,
        likelihood_scan_tail="two_sided",
    ):
        self.poi_name = poi_name
        self.nominal_poi_value = float(nominal_poi_value)
        self.poi_bounds = tuple(poi_bounds)
        self.nuisance_parameter_specs = OrderedDict(
            (spec.name, spec) for spec in nuisance_parameter_specs
        )
        self.data_values = np.asarray(data_values, dtype=float)
        self.data_errors = np.asarray(data_errors, dtype=float)
        self.build_expected_values = build_expected_values
        self.likelihood_scan_points = max(int(likelihood_scan_points), 3)
        self.last_likelihood_scan = None
        # The scan can report multiple confidence intervals
        self.likelihood_scan_intervals = self.resolve_likelihood_scan_configuration(
            likelihood_scan_thresholds,
            likelihood_scan_confidence_levels,
            likelihood_scan_tail,
        )

        self.floating_np_names = [
            name for name, spec in self.nuisance_parameter_specs.items() if spec.fit
        ]
        self.prefit = FitParameterSnapshot(
            pois=OrderedDict([(self.poi_name, self.nominal_poi_value)]),
            nuisance_parameters=OrderedDict(
                (name, spec.initial_value)
                for name, spec in self.nuisance_parameter_specs.items()
            ),
        )

        if uncertainty_method=='Second_Derivative':
            self.estimate_poi_uncertainty = self.uncertainty_from_second_derivative
        elif uncertainty_method=='Likelihood_Scan':
            self.estimate_poi_uncertainty = self.uncertainty_from_likelihood_scan
        else:
            raise ValueError(f"Unknown uncertainty method: {uncertainty_method}")

    @staticmethod
    def parse_confidence_level(confidence_level):
        if confidence_level is None:
            return None
        if isinstance(confidence_level, str):
            confidence_level = confidence_level.strip()
            if confidence_level.endswith("%"):
                return float(confidence_level[:-1]) / 100.0
        confidence_level = float(confidence_level)
        if confidence_level > 1.0:
            confidence_level /= 100.0
        if not 0.0 < confidence_level < 1.0:
            raise ValueError(
                "likelihood_scan_confidence_levels values must be between 0 and 1, "
                "or between 0 and 100 when given as a percentage."
            )
        return confidence_level

    @staticmethod
    def ensure_config_list(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple, set, np.ndarray)):
            return list(value)
        return [value]

    @classmethod
    def make_interval_spec_from_threshold(cls, threshold):
        resolved_threshold = float(threshold)
        if resolved_threshold <= 0:
            raise ValueError("likelihood_scan_thresholds values must be positive.")
        return LikelihoodIntervalResult(
            label=f"Delta(-2lnL)={resolved_threshold:.3f}",
            threshold_delta_neg2_log_likelihood=resolved_threshold,
            confidence_level=None,
            tail_probability_definition="custom_threshold",
        )

    @classmethod
    def make_interval_spec_from_confidence_level(cls, confidence_level, tail):
        resolved_confidence_level = cls.parse_confidence_level(confidence_level)
        resolved_tail = str(tail).strip().lower()
        if resolved_tail not in {"two_sided", "one_sided"}:
            raise ValueError(
                "likelihood_scan_tail must be either 'two_sided' or 'one_sided'."
            )
        if resolved_tail == "two_sided":
            resolved_threshold = float(chi2.ppf(resolved_confidence_level, df=1))
        else:
            if resolved_confidence_level <= 0.5:
                raise ValueError(
                    "one-sided likelihood scan confidence levels must be above 0.5."
                )
            resolved_threshold = float(
                chi2.ppf(2.0 * resolved_confidence_level - 1.0, df=1)
            )
        return LikelihoodIntervalResult(
            label=f"{100.0 * resolved_confidence_level:.1f}% CL",
            threshold_delta_neg2_log_likelihood=resolved_threshold,
            confidence_level=resolved_confidence_level,
            tail_probability_definition=resolved_tail,
        )

    @classmethod
    def resolve_likelihood_scan_configuration(
        cls,
        thresholds,
        confidence_levels,
        tail,
    ):
        interval_specs = [
            LikelihoodIntervalResult(
                label="1 sigma",
                threshold_delta_neg2_log_likelihood=ONE_SIGMA_DELTA_NEG2_LOG_LIKELIHOOD,
                confidence_level=ONE_SIGMA_CONFIDENCE_LEVEL,
                tail_probability_definition="two_sided",
            )
        ]
        interval_specs.extend(
            cls.make_interval_spec_from_confidence_level(level, tail)
            for level in cls.ensure_config_list(confidence_levels)
        )
        interval_specs.extend(
            cls.make_interval_spec_from_threshold(scan_threshold)
            for scan_threshold in cls.ensure_config_list(thresholds)
        )

        deduplicated_specs = OrderedDict()
        for spec in interval_specs:
            dedup_key = (
                round(spec.threshold_delta_neg2_log_likelihood, 12),
                spec.tail_probability_definition,
            )
            if dedup_key not in deduplicated_specs:
                deduplicated_specs[dedup_key] = spec
        return tuple(deduplicated_specs.values())

    def fit(self):
        x0 = [self.nominal_poi_value]
        bounds = [self.poi_bounds]
        for name in self.floating_np_names:
            spec = self.nuisance_parameter_specs[name]
            x0.append(float(spec.initial_value))
            bounds.append(tuple(spec.bounds))

        result = minimize(
            self.objective,
            x0=np.asarray(x0, dtype=float),
            bounds=bounds,
            method="L-BFGS-B",
        )
        poi_value, nuisance_values = self.unpack_parameters(result.x)
        poi_uncertainty_up, poi_uncertainty_down = self.estimate_poi_uncertainty(poi_value, nuisance_values)
        postfit = FitParameterSnapshot(
            pois=OrderedDict([(self.poi_name, poi_value)]),
            nuisance_parameters=nuisance_values,
        )
        return SinglePOIFitResult(
            prefit=self.prefit,
            postfit=postfit,
            poi_uncertainty_up=poi_uncertainty_up,
            poi_uncertainty_down=poi_uncertainty_down,
            neg2_log_likelihood=float(result.fun),
            optimizer_result=result,
            likelihood_scan=self.last_likelihood_scan,
        )

    def objective(self, x):
        poi_value, nuisance_values = self.unpack_parameters(x)
        expected_values = self.build_expected_values(poi_value, nuisance_values)
        neg2_log_likelihood = self.poisson_neg2_log_likelihood(
            self.data_values,
            expected_values,
        )

        for name, spec in self.nuisance_parameter_specs.items():
            if spec.constraint_sigma is None or spec.constraint_sigma <= 0:
                continue
            pull = (nuisance_values[name] - spec.initial_value) / spec.constraint_sigma
            neg2_log_likelihood += float(pull**2)
        return neg2_log_likelihood

    @staticmethod
    def poisson_neg2_log_likelihood(observed_values, expected_values):
        observed_values = np.asarray(observed_values, dtype=float)
        expected_values = np.asarray(expected_values, dtype=float)
        if not np.all(np.isfinite(expected_values)):
            return 1e100
        expected_values = np.clip(expected_values, 1e-12, None)

        terms = expected_values - observed_values
        nonzero = observed_values > 0
        terms[nonzero] += observed_values[nonzero] * np.log(
            observed_values[nonzero] / expected_values[nonzero]
        )
        return float(2.0 * np.sum(terms))

    def unpack_parameters(self, x):
        values = np.asarray(x, dtype=float)
        poi_value = float(values[0])
        nuisance_values = OrderedDict(
            (name, float(spec.initial_value))
            for name, spec in self.nuisance_parameter_specs.items()
        )
        for idx, name in enumerate(self.floating_np_names, start=1):
            nuisance_values[name] = float(values[idx])
        return poi_value, nuisance_values

    def profile_nuisance_parameters(self, poi_value, start_nuisance_values=None):
        if len(self.floating_np_names) == 0:
            return float(self.objective([poi_value])), OrderedDict(
                (name, float(spec.initial_value))
                for name, spec in self.nuisance_parameter_specs.items()
            )

        if start_nuisance_values is None:
            start_nuisance_values = OrderedDict(
                (name, self.nuisance_parameter_specs[name].initial_value)
                for name in self.floating_np_names
            )
        x0 = [start_nuisance_values[name] for name in self.floating_np_names]
        bounds = [
            tuple(self.nuisance_parameter_specs[name].bounds)
            for name in self.floating_np_names
        ]

        def np_objective(np_values):
            full_x = [poi_value] + list(np_values)
            return self.objective(full_x)

        result = minimize(
            np_objective,
            x0=np.asarray(x0, dtype=float),
            bounds=bounds,
            method="L-BFGS-B",
        )
        nuisance_values = OrderedDict(
            (name, float(spec.initial_value))
            for name, spec in self.nuisance_parameter_specs.items()
        )
        for idx, name in enumerate(self.floating_np_names):
            nuisance_values[name] = float(result.x[idx])
        return float(result.fun), nuisance_values

    @staticmethod
    def interpolate_crossing(x0, y0, x1, y1, threshold):
        if y0 == y1:
            return 0.5 * (x0 + x1)
        fraction = (threshold - y0) / (y1 - y0)
        fraction = np.clip(fraction, 0.0, 1.0)
        return float(x0 + fraction * (x1 - x0))

    def extract_interval(self, poi_values, delta_neg2_log_likelihood_values, threshold):
        interval_lower = None
        interval_upper = None
        best_index = int(np.argmin(delta_neg2_log_likelihood_values))

        for idx in range(best_index, 0, -1):
            y0 = delta_neg2_log_likelihood_values[idx - 1]
            y1 = delta_neg2_log_likelihood_values[idx]
            if y0 > threshold and y1 <= threshold:
                interval_lower = self.interpolate_crossing(
                    poi_values[idx - 1],
                    y0,
                    poi_values[idx],
                    y1,
                    threshold,
                )
                break
        if interval_lower is None and delta_neg2_log_likelihood_values[0] <= threshold:
            interval_lower = float(poi_values[0])

        for idx in range(best_index, len(poi_values) - 1):
            y0 = delta_neg2_log_likelihood_values[idx]
            y1 = delta_neg2_log_likelihood_values[idx + 1]
            if y0 <= threshold and y1 > threshold:
                interval_upper = self.interpolate_crossing(
                    poi_values[idx],
                    y0,
                    poi_values[idx + 1],
                    y1,
                    threshold,
                )
                break
        if interval_upper is None and delta_neg2_log_likelihood_values[-1] <= threshold:
            interval_upper = float(poi_values[-1])

        return interval_lower, interval_upper

    def build_likelihood_scan(self, best_value, best_nuisance_values):
        total_points = self.likelihood_scan_points
        points_left = max(total_points // 2, 1)
        points_right = max(total_points - points_left - 1, 1)

        left_grid = np.linspace(best_value, self.poi_bounds[0], points_left + 1)
        right_grid = np.linspace(best_value, self.poi_bounds[1], points_right + 1)

        left_points = []
        current_nuisance = OrderedDict(best_nuisance_values)
        for poi_value in left_grid:
            profiled_nll, current_nuisance = self.profile_nuisance_parameters(
                float(poi_value),
                current_nuisance,
            )
            left_points.append((float(poi_value), profiled_nll))

        right_points = []
        current_nuisance = OrderedDict(best_nuisance_values)
        for poi_value in right_grid[1:]:
            profiled_nll, current_nuisance = self.profile_nuisance_parameters(
                float(poi_value),
                current_nuisance,
            )
            right_points.append((float(poi_value), profiled_nll))

        scan_points = OrderedDict()
        for poi_value, profiled_nll in sorted(
            left_points + right_points,
            key=lambda item: item[0],
        ):
            # in case of the best fit value sitting at the boundary
            if poi_value in scan_points:
                scan_points[poi_value] = min(scan_points[poi_value], profiled_nll)
            else:
                scan_points[poi_value] = profiled_nll

        poi_values = np.asarray(list(scan_points.keys()), dtype=float)
        neg2_log_likelihood_values = np.asarray(
            list(scan_points.values()),
            dtype=float,
        )
        delta_neg2_log_likelihood_values = (
            neg2_log_likelihood_values - float(np.min(neg2_log_likelihood_values))
        )

        intervals = []
        for spec in self.likelihood_scan_intervals:
            interval_lower, interval_upper = self.extract_interval(
                poi_values,
                delta_neg2_log_likelihood_values,
                spec.threshold_delta_neg2_log_likelihood,
            )
            intervals.append(
                LikelihoodIntervalResult(
                    label=spec.label,
                    threshold_delta_neg2_log_likelihood=spec.threshold_delta_neg2_log_likelihood,
                    confidence_level=spec.confidence_level,
                    tail_probability_definition=spec.tail_probability_definition,
                    interval_lower=interval_lower,
                    interval_upper=interval_upper,
                )
            )

        one_sigma_interval = next(
            (
                interval
                for interval in intervals
                if np.isclose(
                    interval.threshold_delta_neg2_log_likelihood,
                    ONE_SIGMA_DELTA_NEG2_LOG_LIKELIHOOD,
                )
            ),
            None,
        )

        return LikelihoodScanResult(
            poi_values=tuple(float(value) for value in poi_values),
            neg2_log_likelihood_values=tuple(
                float(value) for value in neg2_log_likelihood_values
            ),
            delta_neg2_log_likelihood_values=tuple(
                float(value) for value in delta_neg2_log_likelihood_values
            ),
            one_sigma_interval=one_sigma_interval,
            intervals=tuple(intervals),
        )

    def uncertainty_from_second_derivative(self, best_value, nuisance_values):
        self.last_likelihood_scan = None
        step = 1e-3 * max(1.0, abs(best_value))
        x_low = max(self.poi_bounds[0], best_value - step)
        x_high = min(self.poi_bounds[1], best_value + step)
        if x_low == best_value or x_high == best_value:
            return 0.0, 0.0

        f0, _ = self.profile_nuisance_parameters(best_value, nuisance_values)
        f_high, _ = self.profile_nuisance_parameters(x_high, nuisance_values)
        f_low, _ = self.profile_nuisance_parameters(x_low, nuisance_values)
        second_derivative = (f_high - 2.0 * f0 + f_low) / (step**2)
        if second_derivative <= 0 or not np.isfinite(second_derivative):
            return 0.0, 0.0
        unc = float(np.sqrt(2.0 / second_derivative))
        return unc, unc

    def uncertainty_from_likelihood_scan(self, best_value, nuisance_values):
        self.last_likelihood_scan = self.build_likelihood_scan(
            best_value,
            nuisance_values,
        )
        lower = self.last_likelihood_scan.interval_lower
        upper = self.last_likelihood_scan.interval_upper
        poi_uncertainty_down = (
            0.0 if lower is None else float(max(best_value - lower, 0.0))
        )
        poi_uncertainty_up = (
            0.0 if upper is None else float(max(upper - best_value, 0.0))
        )
        return poi_uncertainty_up, poi_uncertainty_down
