"""Platform and scenario parameters of the swarm simulation, with provenance.

status: "published" = stated in the cited source; "derived" = computed from
published values; "assumed" = our modelling choice, varied in the sensitivity
study. No value here was measured by us.
"""
from inspectswarm.swarm import Platform

SCOUT_TABLE = [
    # name, value, unit, status, source / reasoning
    ("Airframe envelope", "10 x 10 x 4", "cm", "published", "SmartBee, public posts by Y. Fu (NTU LIFT lab), 2026"),
    ("Sensors", "RGB-D camera, accelerometer", "", "published", "same posts"),
    ("On-board edge AI, self-attachment", "yes", "", "published", "same posts; no performance figures given"),
    ("Take-off mass", 27, "g", "assumed", "Crazyflie 2.1 datasheet value, used as stand-in for an open nano-drone of this size"),
    ("Maximum payload", 15, "g", "assumed", "Crazyflie 2.1 datasheet"),
    ("Battery", "250 mAh, 3.7 V (0.925 Wh)", "", "assumed", "Crazyflie 2.1 stock battery"),
    ("Hover endurance, no payload", 7, "min", "assumed", "Crazyflie 2.1 datasheet"),
    ("Usable endurance with sensor payload", 5, "min", "assumed", "about 30% derating for payload; varied 3 to 7 min"),
    ("Mean flight power", 7.9, "W", "derived", "0.925 Wh / 7 min"),
    ("Speed between zones", 0.5, "m/s", "assumed", "slow close-range flight; varied 0.25 to 1.0"),
    ("Dwell per zone", 2.0, "s", "assumed", "stabilise, capture, infer; inference time measured separately"),
    ("Return reserve", 20, "s", "assumed", ""),
    ("Battery swap", 60, "s", "assumed", "manual or docked swap"),
]

CARRIER_TABLE = [
    ("Class", "about 1.6 kg multirotor with contact arm", "", "published",
     "Zhang et al. 2021: AscTec Firefly, 1.6 kg maximum take-off mass, payload near the limit"),
    ("Endurance", 15, "min", "published", "Zhang et al. 2021 (about 15 min)"),
    ("Usable endurance", 12, "min", "assumed", "20% margin"),
    ("Mean flight power", 250, "W", "assumed", "order-of-magnitude estimate for a 1.6 kg multirotor; affects energy only"),
    ("Speed between zones", 1.0, "m/s", "assumed", ""),
    ("Time per contact point", 20.0, "s", "assumed", "approach, press, acquire, retreat; varied 10 to 40 s"),
    ("Return reserve", 60, "s", "assumed", ""),
    ("Battery swap", 120, "s", "assumed", ""),
]

SCOUT = Platform("scout", speed=0.5, dwell=2.0, endurance=300.0, reserve=20.0, turnaround=60.0, power=7.9)
CARRIER = Platform("carrier", speed=1.0, dwell=20.0, endurance=720.0, reserve=60.0, turnaround=120.0, power=250.0)

# Scenario: one face of a concrete wall or pier, 20 m x 6 m, 0.5 m zones.
SCENARIO = {"nx": 40, "ny": 12, "zone": 0.5}

# Coupling between visible damage and hidden defects. These are assumptions,
# swept in the sensitivity study; they are not estimated from data.
COUPLING = {"prevalence": 0.20, "p_hidden_damaged": 0.30, "p_hidden_clean": 0.03}
