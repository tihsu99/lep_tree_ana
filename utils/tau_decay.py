import numpy as np


DECAY_MODE_IDS = {
    "notau": 0,
    "pi": 1,
    "rho": 2,
    "e": 3,
    "mu": 4,
    "other": 5,
}

ANALYZING_POWERS = np.array([0.0, 1.0, 0.41, -0.33, -0.34, 0.0])
TAU_DECAY_BRANCHING_RATIOS = np.array([0.0, 0.1077, 0.2537, 0.1773, 0.1731, 0.0])
NOMINAL_BC_VALUES = {
    "B_An": 0.0,
    "B_Ar": 0.000054,
    "B_Ak": 0.149663,
    "B_Bn": 0.0,
    "B_Br": 0.000054,
    "B_Bk": 0.149663,
    "C_nn": 0.784523,
    "C_rr": -0.78451,
    "C_kk": 0.999987,
    "C_nr": 0.0,
    "C_nk": 0.0,
    "C_rk": 0.000724757,
    "C_rn": 0.0,
    "C_kn": 0.0,
    "C_kr": 0.000724757,
}


def normalize_signal_name(signal_name: str) -> str:
    return signal_name.replace("Ztautau_", "").lower()


def get_event_category_from_signal_name(signal_name: str) -> int:
    normalized_name = normalize_signal_name(signal_name)
    decay_names = ("pi", "rho", "e", "mu")
    for pos_name in decay_names:
        for neg_name in decay_names:
            if normalized_name == f"{pos_name}{neg_name}":
                return DECAY_MODE_IDS[pos_name] * 10 + DECAY_MODE_IDS[neg_name]
    raise ValueError(f"Unknown signal name: {signal_name}")


def decode_event_category(event_category: int) -> tuple[int, int]:
    category = int(event_category)
    pos_idx = category // 10
    neg_idx = category % 10
    if not (0 <= pos_idx < len(ANALYZING_POWERS) and 0 <= neg_idx < len(ANALYZING_POWERS)):
        raise ValueError(f"Invalid event category: {event_category}")
    return pos_idx, neg_idx


def decode_event_categories(event_categories) -> tuple[np.ndarray, np.ndarray]:
    categories = np.asarray(event_categories, dtype=int)
    pos_idx = categories // 10
    neg_idx = categories % 10
    all_idx = np.concatenate([np.atleast_1d(pos_idx), np.atleast_1d(neg_idx)])
    if not np.all((all_idx >= 0) & (all_idx < len(ANALYZING_POWERS))):
        raise ValueError("Invalid event categories found")
    return pos_idx, neg_idx


def get_analyzing_powers_from_event_category(event_category: int) -> tuple[float, float]:
    pos_idx, neg_idx = decode_event_category(event_category)
    return -ANALYZING_POWERS[pos_idx], ANALYZING_POWERS[neg_idx]


def get_analyzing_powers_from_event_categories(event_categories):
    pos_idx, neg_idx = decode_event_categories(event_categories)
    return -ANALYZING_POWERS[pos_idx], ANALYZING_POWERS[neg_idx]


def get_branching_ratio_from_event_category(event_category: int) -> float:
    pos_idx, neg_idx = decode_event_category(event_category)
    return TAU_DECAY_BRANCHING_RATIOS[pos_idx] * TAU_DECAY_BRANCHING_RATIOS[neg_idx]
