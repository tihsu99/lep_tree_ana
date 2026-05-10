from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize


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
class SinglePOIFitResult:
    prefit: FitParameterSnapshot
    postfit: FitParameterSnapshot
    poi_uncertainty: float
    neg2_log_likelihood: float
    optimizer_result: object

    @property
    def chi2(self):
        return self.neg2_log_likelihood


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
        poi_uncertainty = self.estimate_poi_uncertainty(poi_value, nuisance_values)
        postfit = FitParameterSnapshot(
            pois=OrderedDict([(self.poi_name, poi_value)]),
            nuisance_parameters=nuisance_values,
        )
        return SinglePOIFitResult(
            prefit=self.prefit,
            postfit=postfit,
            poi_uncertainty=poi_uncertainty,
            neg2_log_likelihood=float(result.fun),
            optimizer_result=result,
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

    def estimate_poi_uncertainty(self, best_value, nuisance_values):
        step = 1e-3 * max(1.0, abs(best_value))
        x_low = max(self.poi_bounds[0], best_value - step)
        x_high = min(self.poi_bounds[1], best_value + step)
        if x_low == best_value or x_high == best_value:
            return 0.0

        def profile_at(poi_value):
            if len(self.floating_np_names) == 0:
                return self.objective([poi_value])

            x0 = [nuisance_values[name] for name in self.floating_np_names]
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
            return float(result.fun)

        f0 = profile_at(best_value)
        second_derivative = (profile_at(x_high) - 2.0 * f0 + profile_at(x_low)) / (step**2)
        if second_derivative <= 0 or not np.isfinite(second_derivative):
            return 0.0
        return float(np.sqrt(2.0 / second_derivative))
