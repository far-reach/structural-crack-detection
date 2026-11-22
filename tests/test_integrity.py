"""Guards for split integrity, ultrasonic ground truth and the swarm simulator.

Tests that need datasets are skipped when the data folder is absent, so the
suite runs on a fresh clone.
"""
import numpy as np
import pytest

from inspectswarm import swarm as S
from inspectswarm import ultrasonic as U
from inspectswarm.paths import CACHE, SDNET


# ------------------------------------------------------------------ vision

@pytest.mark.skipif(not SDNET.exists(), reason="SDNET2018 not present")
def test_grouped_protocol_has_no_parent_leak_and_no_augmented_eval():
    from experiments.leak_study import grouped_protocol, parent_of

    tr, va, _ = grouped_protocol(seed=0)
    sd = ("crack", "intact")
    train_parents = {parent_of(p) for p, c in tr if c in sd}
    assert train_parents, "no SDNET training parents found"
    assert not any(parent_of(p) in train_parents for p, c in va if c in sd)
    assert not any(p.name.startswith("aug_") for p, _ in va)
    assert not {p for p, _ in tr} & {p for p, _ in va}


@pytest.mark.skipif(not (CACHE / "codebrim_test.npz").exists(), reason="CODEBRIM cache not present")
def test_codebrim_splits_do_not_share_source_images():
    def parents(split):
        return {str(n).split("_crop")[0] for n in np.load(CACHE / f"codebrim_{split}.npz")["names"]}

    tr, va, te = parents("train"), parents("val"), parents("test")
    assert not tr & te and not tr & va and not va & te


def test_multilabel_metrics_exact_match():
    from inspectswarm.vision import multilabel_metrics

    y = np.array([[1, 0], [0, 0], [1, 1], [0, 1]])
    p = np.array([[.9, .1], [.2, .6], [.8, .7], [.1, .9]])
    m = multilabel_metrics(y, p, ("a", "b"))
    assert m["multi_target_acc"] == 0.75
    assert m["false_alarm_rate"] == 1.0 and m["damage_recall"] == 1.0


# -------------------------------------------------------------- ultrasonic

def test_thickness_map_follows_published_steps():
    t, clear = U.thickness_map("pk266", edge_margin_mm=60)
    assert t[10, 40] == 569.9 and t[70, 40] == 448.2 and t[120, 40] == 329.8 and t[190, 40] == 210.3
    assert not clear[50, 40] and clear[25, 40]


def test_object_labels_geometry():
    lab, cover = U.object_labels("pk266")
    assert lab[25, 40] == 1 and cover[25, 40] == 244.5      # above the duct at X = 250 mm
    assert lab[50, 40] == 0 and lab[33, 40] == -1            # plain concrete; ambiguous band
    lab, cover = U.object_labels("pk401")
    assert lab[125, 50] == 1 and cover[125, 50] == 150.0     # cuboid at X = 1250, Y = 498.2 mm
    assert lab[125, 20] == 0
    assert (U.object_labels("pk050")[0] == 0).all()


def test_velocity_fit_recovers_known_values():
    d = np.repeat([200.0, 300.0, 500.0], 50)
    t = 20.0 + 2e3 * d / 2700.0 + np.random.default_rng(0).normal(0, 0.5, d.size)
    c, t0 = U.fit_velocity(t, d)
    assert abs(c - 2700) < 20 and abs(t0 - 20) < 2
    assert np.allclose(U.thickness_from_time(20.0 + 2e3 * 300 / 2700, 2700, 20.0), 300)


def test_backwall_pick_on_synthetic_echo():
    t = np.arange(4000) / U.FS * 1e6
    sig = np.exp(-((t - 30) / 8) ** 2) + 0.3 * np.exp(-((t - 400) / 10) ** 2) + 0.1 * np.exp(-((t - 770) / 10) ** 2)
    assert abs(U.pick_backwall(sig[None, None, :], gate_us=110, max_us=900)[0, 0] - 400) < 5


# ------------------------------------------------------------------- swarm

def _world(rng, n_pool=200):
    return S.make_world(rng, 20, 6, 0.5, prevalence=0.2, p_hidden_damaged=0.3, p_hidden_clean=0.03,
                        vis_prob_damaged=rng.beta(5, 1, n_pool), vis_prob_clean=rng.beta(1, 6, n_pool),
                        ut_prob_pos=rng.beta(5, 1, n_pool), ut_prob_neg=rng.beta(1, 6, n_pool))


def test_boustrophedon_visits_every_zone_once():
    p = S.boustrophedon(7, 5)
    assert sorted(p.tolist()) == list(range(35))
    assert all(abs(p[i] % 7 - p[i + 1] % 7) + abs(p[i] // 7 - p[i + 1] // 7) == 1 for i in range(34))


def test_full_coverage_and_speedup_with_more_scouts():
    rng = np.random.default_rng(0)
    w = _world(rng)
    plat = S.Platform("scout", 0.5, 2.0, 300, 20, 60, 7.9)
    path = S.boustrophedon(w.nx, w.ny)
    r1 = S.run_tier(w, S.split_path(path, 1), plat, rng)
    r4 = S.run_tier(w, S.split_path(path, 4), plat, rng)
    assert not np.isnan(r1["visit"]).any() and not np.isnan(r4["visit"]).any()
    assert r4["time"] < r1["time"]
    assert abs(r1["energy_wh"] - r1["air_s"] * 7.9 / 3600) < 1e-9


def test_no_sortie_exceeds_endurance():
    rng = np.random.default_rng(1)
    w = _world(rng)
    plat = S.Platform("scout", 0.5, 2.0, 120, 10, 30, 7.9)
    visit, t, air, sorties, lost = S.fly(w, S.boustrophedon(w.nx, w.ny), plat, rng)
    assert not lost and air <= sorties * plat.endurance + 1e-6


def test_lost_vehicle_work_is_reassigned():
    rng = np.random.default_rng(2)
    w = _world(rng)
    plat = S.Platform("scout", 0.5, 2.0, 80, 5, 60, 7.9, p_loss=1.0)  # every sortie ends in a loss at a random time
    r = S.run_tier(w, S.split_path(S.boustrophedon(w.nx, w.ny), 6), plat, rng)
    assert r["lost"] >= 1
    assert r["lost"] == 6 or not np.isnan(r["visit"]).any()


def test_fusion_moves_posterior_in_the_right_direction():
    prior = np.array([0.1, 0.1, 0.1])
    post = S.fuse(prior, np.array([[0.9, 0.9], [0.5, 0.5], [0.1, 0.1]]), ut_base_rate=0.5)
    assert post[0] > 0.1 and abs(post[1] - 0.1) < 1e-9 and post[2] < 0.1
    assert S.hidden_prior(np.array([1.0, 0.0]), 0.3, 0.03).tolist() == [0.3, 0.03]
