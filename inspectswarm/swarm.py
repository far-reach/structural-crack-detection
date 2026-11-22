"""Two-tier "screen, then confirm" inspection swarm: mission-level simulator.

This is a kinematic and energy-budget simulation, not a flight-dynamics one.
Every platform number is a parameter of `Platform`; the defaults used in the
paper are listed, with their sources or marked as assumed, in
experiments/swarm_params.py.

Tier 1 (scouts) images every zone once and runs the vision classifier.
Tier 2 (contact nodes) visits selected zones and takes point-contact
ultrasonic measurements. Classifier outputs are not simulated: each zone is
bound to a real held-out image and a real held-out A-scan, and the simulator
replays the trained models' outputs on them.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Platform:
    name: str
    speed: float          # m/s between zones
    dwell: float          # s spent per zone (stabilise, sense, infer)
    endurance: float      # s of usable flight per battery
    reserve: float        # s kept in hand when deciding to return
    turnaround: float     # s on the ground for a battery swap
    power: float          # W, mean electrical power in flight
    p_loss: float = 0.0   # probability that a sortie ends with the vehicle lost


@dataclass
class World:
    nx: int
    ny: int
    zone: float                 # zone edge length, m
    surface_damage: np.ndarray  # (n,) bool, ground truth of visible damage
    hidden_defect: np.ndarray   # (n,) bool, ground truth of subsurface anomaly
    p_vision: np.ndarray        # (n,) vision probability of "any damage"
    p_ut: np.ndarray            # (n, k) ultrasonic probability per contact point

    @property
    def n(self):
        return self.nx * self.ny

    def xy(self, idx):
        idx = np.asarray(idx)
        return np.stack([(idx % self.nx + 0.5) * self.zone, (idx // self.nx + 0.5) * self.zone], -1)


def make_world(rng, nx, ny, zone, *, prevalence, p_hidden_damaged, p_hidden_clean,
               vis_prob_damaged, vis_prob_clean, ut_prob_pos, ut_prob_neg, k_contacts=3,
               cluster=0.0) -> World:
    """Draw a structure. `vis_prob_*` and `ut_prob_*` are pools of real model
    outputs on held-out samples of each true state; zones sample from them.
    `cluster` in [0,1) makes damage spatially clustered (smoothed random field)."""
    n = nx * ny
    field = rng.random((ny, nx))
    if cluster > 0:
        from scipy.ndimage import gaussian_filter

        field = gaussian_filter(rng.standard_normal((ny, nx)), sigma=1 + 4 * cluster, mode="reflect")
        field = field.argsort(None).argsort().reshape(ny, nx) / (n - 1)
    damaged = (field < prevalence).ravel()
    hidden = rng.random(n) < np.where(damaged, p_hidden_damaged, p_hidden_clean)
    p_vis = np.where(damaged, rng.choice(vis_prob_damaged, n), rng.choice(vis_prob_clean, n))
    p_ut = np.where(hidden[:, None], rng.choice(ut_prob_pos, (n, k_contacts)), rng.choice(ut_prob_neg, (n, k_contacts)))
    return World(nx, ny, zone, damaged, hidden, p_vis, p_ut)


def boustrophedon(nx, ny) -> np.ndarray:
    rows = [np.arange(nx) if r % 2 == 0 else np.arange(nx)[::-1] for r in range(ny)]
    return np.concatenate([r * nx + row for r, row in enumerate(rows)])


def split_path(path, n_agents):
    return [p for p in np.array_split(path, n_agents) if len(p)]


def nearest_neighbour_tour(xy, start):
    left = list(range(len(xy)))
    tour, cur = [], np.asarray(start, float)
    while left:
        d = np.linalg.norm(xy[left] - cur, axis=1)
        j = left.pop(int(d.argmin()))
        tour.append(j)
        cur = xy[j]
    return np.array(tour, int)


def fly(world: World, zones, plat: Platform, rng, base=(0.0, 0.0), t_start=0.0):
    """Serve `zones` in order with battery-limited sorties from `base`.
    Returns visit time per zone (nan if never served), the mission end time,
    airborne seconds, number of sorties and whether the vehicle was lost."""
    base = np.asarray(base, float)
    xy = world.xy(zones) if len(zones) else np.zeros((0, 2))
    visit = np.full(len(zones), np.nan)
    t, air, sorties, i, lost = t_start, 0.0, 0, 0, False
    while i < len(zones):
        sorties += 1
        lose_at = rng.random() * plat.endurance if rng.random() < plat.p_loss else np.inf
        used, pos = 0.0, base
        while i < len(zones):
            leg = np.linalg.norm(xy[i] - pos) / plat.speed
            back = np.linalg.norm(xy[i] - base) / plat.speed
            if used + leg + plat.dwell + back + plat.reserve > plat.endurance:
                break
            if used + leg + plat.dwell > lose_at:
                lost = True
                break
            used += leg + plat.dwell
            visit[i] = t + used
            pos = xy[i]
            i += 1
        if lost:
            air += min(lose_at, used)
            t += min(lose_at, used)
            break
        if used == 0.0:  # zone unreachable within one battery
            break
        used += np.linalg.norm(pos - base) / plat.speed
        air += used
        t += used + (plat.turnaround if i < len(zones) else 0.0)
    return visit, t, air, sorties, lost


def run_tier(world, assignments, plat, rng, base=(0.0, 0.0), t_start=0.0):
    """Fly several agents in parallel. Zones left unserved by a lost vehicle
    are handed to the surviving agent that finishes first, one round at a time."""
    n_agents = len(assignments)
    visit = np.full(world.n, np.nan)
    t_end = np.full(n_agents, t_start)
    alive = np.ones(n_agents, bool)
    air, sorties = 0.0, 0
    queue = [(a, np.asarray(z)) for a, z in enumerate(assignments)]
    while queue:
        a, zones = queue.pop(0)
        v, t, f, s, lost = fly(world, zones, plat, rng, base, t_end[a])
        visit[zones[~np.isnan(v)]] = v[~np.isnan(v)]
        t_end[a], air, sorties = t, air + f, sorties + s
        left = zones[np.isnan(v)]
        if lost:
            alive[a] = False
        if len(left) and alive.any() and (lost or len(left) < len(zones)):
            heir = int(np.where(alive, t_end, np.inf).argmin())
            queue.append((heir, left))
    return {"visit": visit, "time": float(t_end.max()) - t_start, "air_s": air, "sorties": sorties,
            "lost": int((~alive).sum()), "energy_wh": air * plat.power / 3600.0}


# ------------------------------------------------------------------ fusion

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def hidden_prior(p_vision, p_hidden_damaged, p_hidden_clean):
    """P(hidden defect) for a zone, marginalised over the vision evidence."""
    return p_vision * p_hidden_damaged + (1 - p_vision) * p_hidden_clean


def fuse(prior, p_ut, ut_base_rate):
    """Posterior of a hidden defect from the vision-informed prior and k
    ultrasonic contact readings. Each reading is a classifier probability
    trained at prevalence `ut_base_rate`; dividing its odds by the training
    odds gives a likelihood ratio (readings treated as independent)."""
    lr = logit(p_ut) - logit(ut_base_rate)
    post = logit(prior) + np.nansum(lr, axis=-1)
    return 1 / (1 + np.exp(-post))


def detection_metrics(truth, decision):
    truth, decision = np.asarray(truth, bool), np.asarray(decision, bool)
    tp, fp = int((truth & decision).sum()), int((~truth & decision).sum())
    fn = int((truth & ~decision).sum())
    rec = tp / (tp + fn) if tp + fn else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return {"recall": rec, "precision": prec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
