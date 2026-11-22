**Source table.** The v1 recipe retrained under two protocols (three seeds, mean ± standard deviation).

| Protocol | Accuracy (%) | Macro F1 | Accuracy on crack/intact source (%) | Accuracy on corrosion/spalling source (%) |
|---|---|---|---|---|
| Original tile split (leaky) | 83.6 ± 0.7 | 0.826 ± 0.006 | 84.7 ± 0.9 | 80.3 ± 0.7 |
| Grouped split, originals only | 83.2 ± 1.4 | 0.813 ± 0.017 | 83.0 ± 1.3 | 83.7 ± 1.7 |

tile_model_on_tile_val per-class F1 mean: {'corrosion': 0.782, 'crack': 0.83, 'intact': 0.855, 'spalling': 0.836}  n = 1600
grouped_model_on_grouped_val per-class F1 mean: {'corrosion': 0.887, 'crack': 0.815, 'intact': 0.842, 'spalling': 0.707}  n = 1551

**Source table.** CODEBRIM test results, mean ± standard deviation over seeds (%).

| Model | Parameters (M) | Input (px) | Distilled | Seeds | Multi-target accuracy | Macro F1 | Damage recall | False alarms on background |
|---|---|---|---|---|---|---|---|---|
| ConvNeXt-Tiny | 27.82 | 224 | no | 3 | 80.2 ± 0.9 | 88.7 ± 0.5 | 95.5 ± 1.8 | 1.9 ± 0.0 |
| ResNet50 | 23.52 | 224 | no | 3 | 75.8 ± 1.3 | 86.6 ± 0.4 | 96.0 ± 0.9 | 5.2 ± 1.9 |
| MobileNetV3-Small | 1.52 | 224 | no | 3 | 76.9 ± 0.7 | 87.0 ± 0.6 | 96.4 ± 0.2 | 7.1 ± 1.9 |
| MobileNetV3-Small | 1.52 | 224 | yes | 3 | 79.3 ± 0.8 | 87.9 ± 0.2 | 95.8 ± 0.5 | 4.5 ± 0.9 |
| NanoNet-32 | 0.14 | 128 | no | 3 | 69.8 ± 1.7 | 81.7 ± 1.0 | 90.1 ± 0.4 | 9.1 ± 1.4 |
| NanoNet-32 | 0.14 | 128 | yes | 3 | 70.0 ± 0.7 | 81.1 ± 0.3 | 89.1 ± 1.0 | 8.2 ± 1.1 |
| NanoNet-16 | 0.04 | 96 | no | 3 | 66.8 ± 1.8 | 78.8 ± 0.9 | 88.1 ± 0.6 | 13.4 ± 3.5 |
| NanoNet-16 | 0.04 | 96 | yes | 3 | 67.8 ± 0.1 | 79.4 ± 0.4 | 87.7 ± 0.4 | 10.0 ± 0.3 |

**Source table.** Per-class F1 on the CODEBRIM test set (teacher and distilled students, mean over seeds).

| Model | Crack | Spallation | Efflorescence | ExposedBars | CorrosionStain |
|---|---|---|---|---|---|
| ConvNeXt-Tiny | 0.911 | 0.856 | 0.883 | 0.951 | 0.835 |
| MobileNetV3-Small | 0.899 | 0.850 | 0.861 | 0.935 | 0.851 |
| NanoNet-16 | 0.758 | 0.804 | 0.716 | 0.904 | 0.786 |
| NanoNet-32 | 0.775 | 0.819 | 0.755 | 0.919 | 0.789 |

**Source table.** Edge budget of the seed-0 models: file size, multi-target accuracy and median single-image latency on one laptop CPU thread, for 32-bit float and 8-bit integer versions.

| Model | Input (px) | Size float (MB) | Size 8-bit (MB) | Accuracy float (%) | Accuracy 8-bit (%) | Latency float (ms) | Latency 8-bit (ms) |
|---|---|---|---|---|---|---|---|
| ConvNeXt-Tiny | 224 | 106.21 | 27.26 | 81.2 | 45.6 | 55.3 | 79.6 |
| ResNet50 | 224 | 89.64 | 22.91 | 74.1 | 73.1 | 37.4 | 29.3 |
| MobileNetV3-Small | 224 | 5.82 | 1.76 | 77.2 | 40.0 | 1.4 | 2.8 |
| MobileNetV3-Small (distilled) | 224 | 5.82 | 1.76 | 78.8 | 30.9 | 1.4 | 2.8 |
| NanoNet-32 | 128 | 0.54 | 0.19 | 70.7 | 70.7 | 0.4 | 0.4 |
| NanoNet-32 (distilled) | 128 | 0.54 | 0.19 | 69.3 | 68.7 | 0.4 | 0.4 |
| NanoNet-16 | 96 | 0.15 | 0.08 | 68.4 | 68.0 | 0.1 | 0.2 |
| NanoNet-16 (distilled) | 96 | 0.15 | 0.08 | 67.7 | 68.0 | 0.1 | 0.2 |

**Cross-source A** (CODEBRIM models on SDNET2018 / Raidathmane originals), n = {'corrosion': 267, 'crack': 600, 'intact': 600, 'spalling': 84} 

| Model | damage_recall | false_alarm_rate_intact | balanced_acc_damage_vs_intact | flag_recall_crack | class_recall_crack | flag_recall_corrosion | class_recall_corrosion | flag_recall_spalling | class_recall_spalling |
|---|---|---|---|---|---|---|---|---|---|
| convnext_tiny_r224_s0 | 63.1 | 10.2 | 76.5 | 53.7 | 49.7 | 77.2 | 39.3 | 85.7 | 39.3 |
| resnet50_r224_s0 | 77.1 | 21.3 | 77.9 | 70.0 | 60.7 | 87.3 | 44.6 | 95.2 | 52.4 |
| mobilenetv3_small_100_r224_s0 | 59.6 | 12.2 | 73.7 | 45.8 | 38.8 | 80.9 | 43.8 | 90.5 | 31.0 |
| mobilenetv3_small_100_r224_kd-convnext_tiny_r224_s0_s0 | 65.3 | 18.8 | 73.2 | 53.2 | 47.5 | 84.6 | 41.9 | 90.5 | 42.9 |
| nano32_r128_s0 | 57.3 | 23.7 | 66.8 | 42.3 | 34.8 | 83.5 | 39.7 | 81.0 | 59.5 |
| nano32_r128_kd-convnext_tiny_r224_s0_s0 | 56.4 | 23.5 | 66.4 | 40.5 | 30.0 | 83.5 | 42.3 | 83.3 | 56.0 |
| nano16_r96_s0 | 54.6 | 29.8 | 62.4 | 40.7 | 28.0 | 79.4 | 37.8 | 75.0 | 51.2 |
| nano16_r96_kd-convnext_tiny_r224_s0_s0 | 54.0 | 33.2 | 60.4 | 39.8 | 27.3 | 76.8 | 37.1 | 83.3 | 61.9 |

**Cross-source B** (v1.0 classifier on CODEBRIM test): {
"intact_recall_on_background": 0.19480519480519481,
"n_background": 154,
"recall_corrosion": 0.16363636363636364,
"flagged_as_damage_corrosion": 0.9818181818181818,
"n_corrosion": 55,
"recall_crack": 0.16535433070866143,
"flagged_as_damage_crack": 0.9606299212598425,
"n_crack": 127,
"recall_spalling": 0.17647058823529413,
"flagged_as_damage_spalling": 1.0,
"n_spalling": 17,
"damage_recall_all_defect_crops": 0.9728033472803347,
"balanced_acc_damage_vs_intact": 0.5838042710427648,
"prediction_share": {
"corrosion": 0.5490506329113924,
"crack": 0.04272151898734177,
"intact": 0.0680379746835443,
"spalling": 0.3401898734177215
}
} 

**Source table.** Embedded-object detection from single A-scans, cross-specimen (three seeds).

| Wave, orientation | AUROC ducts (Pk266) | F1 ducts | AUROC cuboids (Pk401) | F1 cuboids | Baseline AUROC ducts | Baseline AUROC cuboids | Specificity Pk266 (%) | Specificity Pk401 (%) | Specificity plain Pk050 (%) |
|---|---|---|---|---|---|---|---|---|---|
| Longitudinal, 00° | 0.866 ± 0.017 | 0.580 ± 0.028 | 0.878 ± 0.004 | 0.314 ± 0.009 | 0.612 | 0.641 | 96.9 ± 0.4 | 74.4 ± 1.2 | 94.9 ± 0.1 |
| Longitudinal, 90° | 0.848 ± 0.014 | 0.567 ± 0.021 | 0.853 ± 0.009 | 0.315 ± 0.010 | 0.615 | 0.630 | 97.0 ± 0.2 | 76.9 ± 1.8 | 90.8 ± 0.2 |
| Shear, 00° | 0.850 ± 0.001 | 0.692 ± 0.024 | 0.978 ± 0.002 | 0.709 ± 0.016 | 0.869 | 0.919 | 99.6 ± 0.1 | 96.6 ± 0.1 | 97.7 ± 0.1 |
| Shear, 90° | 0.961 ± 0.002 | 0.901 ± 0.003 | 0.993 ± 0.002 | 0.806 ± 0.014 | 0.806 | 0.944 | 99.1 ± 0.2 | 97.1 ± 0.1 | 99.1 ± 0.2 |

detect_Long_Rot00 {'pk266': {'64.4': 0.59, '121.4': 0.51, '184.6': 0.21, '244.5': 0.47}, 'pk401': {'90.0': 0.75, '150.0': 0.81, '210.0': 0.92, '270.0': 0.89}} recall {'pk266': 0.445, 'pk401': 0.843} precision {'pk266': 0.834, 'pk401': 0.193} uncal F1 {'pk266': 0.483, 'pk401': 0.349}
detect_Long_Rot90 {'pk266': {'64.4': 0.74, '121.4': 0.26, '184.6': 0.4, '244.5': 0.32}, 'pk401': {'90.0': 0.58, '150.0': 0.88, '210.0': 0.89, '270.0': 0.78}} recall {'pk266': 0.429, 'pk401': 0.783} precision {'pk266': 0.837, 'pk401': 0.197} uncal F1 {'pk266': 0.433, 'pk401': 0.342}
detect_Shear_Rot00 {'pk266': {'64.4': 0.01, '121.4': 0.82, '184.6': 0.81, '244.5': 0.5}, 'pk401': {'90.0': 0.32, '150.0': 0.96, '210.0': 0.96, '270.0': 0.99}} recall {'pk266': 0.536, 'pk401': 0.808} precision {'pk266': 0.978, 'pk401': 0.631} uncal F1 {'pk266': 0.646, 'pk401': 0.697}
detect_Shear_Rot90 {'pk266': {'64.4': 0.77, '121.4': 0.92, '184.6': 0.84, '244.5': 0.83}, 'pk401': {'90.0': 0.94, '150.0': 0.9, '210.0': 0.96, '270.0': 0.97}} recall {'pk266': 0.84, 'pk401': 0.943} precision {'pk266': 0.972, 'pk401': 0.705} uncal F1 {'pk266': 0.859, 'pk401': 0.815}

**Source table.** Thickness from single A-scans with the wave speed calibrated on Pk050 only.

| Wave, orientation | c from Pk050 (m/s) | Test specimen | A-scans | Median error (mm) | Mean error (mm) | Within 10 mm (%) | Within 5% (%) | Median error, 5 contacts (mm) | Within 5%, 5 contacts (%) |
|---|---|---|---|---|---|---|---|---|---|
| Shear, 00° | 2773 | Pk266 | 3266 | 3.3 | 20.9 | 69 | 77 | 2.9 | 80 |
| Shear, 00° | 2773 | Pk401 | 6534 | 8.1 | 22.6 | 59 | 86 | 7.4 | 89 |
| Shear, 90° | 2790 | Pk266 | 3266 | 3.8 | 16.3 | 85 | 91 | 3.4 | 94 |
| Shear, 90° | 2790 | Pk401 | 6534 | 9.0 | 31.7 | 56 | 87 | 8.6 | 90 |
| Longitudinal, 00° | 4439 | Pk266 | 3266 | 6.5 | 83.2 | 57 | 65 | 5.0 | 68 |
| Longitudinal, 00° | 4439 | Pk401 | 6534 | 105.9 | 145.5 | 29 | 32 | 112.7 | 33 |
| Longitudinal, 90° | 4436 | Pk266 | 3266 | 6.9 | 81.5 | 58 | 66 | 5.0 | 70 |
| Longitudinal, 90° | 4436 | Pk401 | 6534 | 109.7 | 148.5 | 31 | 33 | 88.9 | 36 |

Pk050 step 2 check: {'Shear_Rot00': 450.7, 'Shear_Rot90': 451.9, 'Long_Rot00': 451.3, 'Long_Rot90': 450.5} 

### swarm_convnext_tiny_r224_s0__Shear_Rot90

tier accuracy: {"vision_damage_recall": 0.975, "vision_false_alarm": 0.026, "ut_recall_single_contact": 0.874, "ut_false_alarm_single_contact": 0.018}

**Source table.** Tier 1 coverage of 480 zones against the number of scouts.

| Scouts | Time (min) | Sorties | Flight energy (Wh) | 5% loss: time (min) / coverage (%) | 10% loss: time / coverage | 20% loss: time / coverage |
|---|---|---|---|---|---|---|
| 1 | 31.9 | 6 | 3.54 | 29.1 / 91.5 | 25.8 / 81.3 | 20.0 / 63.8 |
| 2 | 15.1 | 6 | 3.43 | 16.3 / 100.0 | 17.9 / 99.6 | 19.1 / 95.1 |
| 4 | 8.9 | 8 | 3.91 | 9.4 / 100.0 | 10.1 / 100.0 | 12.1 / 100.0 |
| 8 | 4.0 | 8 | 3.92 | 4.4 / 100.0 | 5.0 / 100.0 | 5.6 / 100.0 |
| 12 | 2.9 | 12 | 4.35 | 3.1 / 100.0 | 3.6 / 100.0 | 4.3 / 100.0 |
| 16 | 2.4 | 16 | 4.65 | 2.7 / 100.0 | 3.0 / 100.0 | 3.6 / 100.0 |

**Source table.** Confirmation policies (mean ± standard deviation over simulated structures).

| Policy | Contacts | Tier 2 time (min) | Tier 2 energy (Wh) | Hidden-defect recall (%) | Precision (%) | Recall at threshold 0.2 (%) | Precision at threshold 0.2 (%) |
|---|---|---|---|---|---|---|---|
| Vision only | 0 | 0 | 0 | 0 ± 0 | nan ± nan | 69 | 29 |
| Random zones, same budget | 310 | 129 | 454 | 20 ± 6 | 100 ± 2 | 75 | 37 |
| Highest prior, half budget | 154 | 64 | 227 | 36 ± 6 | 100 ± 0 | 67 | 47 |
| Flagged zones | 310 | 128 | 453 | 67 ± 7 | 100 ± 0 | 68 | 100 |
| Flagged + 10% audit | 422 | 175 | 615 | 69 ± 7 | 100 ± 0 | 70 | 100 |
| Exhaustive | 1440 | 595 | 2088 | 94 ± 4 | 100 ± 0 | 96 | 100 |
| Perfect scout (oracle) | 288 | 119 | 419 | 68 ± 7 | 100 ± 0 | 69 | 92 |

n_hidden mean: 40.8  tier1 time: 4.0 min, tier1 energy: 3.92 Wh

Sensitivity: k_contacts (recall % / precision % / contacts)

| k_contacts | flagged | random same budget | exhaustive |
|---|---|---|---|
| 1 | 60 / 97 / 103 | 18 / 96 / 103 | 80 / 94 / 480 |
| 2 | 63 / 100 / 206 | 20 / 99 / 206 | 87 / 99 / 960 |
| 3 | 67 / 100 / 308 | 20 / 99 / 308 | 95 / 100 / 1440 |
| 5 | 70 / 100 / 514 | 22 / 100 / 514 | 99 / 100 / 2400 |

Sensitivity: ut_noise (recall % / precision % / contacts)

| ut_noise | flagged | random same budget | exhaustive |
|---|---|---|---|
| 0.0 | 67 / 100 / 308 | 20 / 99 / 308 | 95 / 100 / 1440 |
| 0.2 | 65 / 100 / 308 | 19 / 100 / 308 | 88 / 100 / 1440 |
| 0.4 | 61 / 100 / 308 | 17 / 100 / 308 | 76 / 99 / 1440 |
| 0.6 | 52 / 98 / 308 | 13 / 96 / 308 | 60 / 97 / 1440 |
| 0.8 | 39 / 67 / 308 | 9 / 67 / 308 | 44 / 66 / 1440 |
| 1.0 | 27 / 29 / 308 | 7 / 15 / 308 | 31 / 14 / 1440 |

Sensitivity: lift (recall % / precision % / contacts)

| lift | flagged | random same budget | exhaustive |
|---|---|---|---|
| 0.3_0.3 | 21 / 100 / 308 | 21 / 100 / 308 | 95 / 100 / 1440 |
| 0.3_0.1 | 41 / 100 / 308 | 21 / 100 / 308 | 94 / 100 / 1440 |
| 0.3_0.03 | 67 / 100 / 308 | 20 / 99 / 308 | 95 / 100 / 1440 |
| 0.3_0.01 | 84 / 100 / 308 | 20 / 99 / 308 | 95 / 100 / 1440 |
| 0.5_0.01 | 88 / 100 / 308 | 38 / 68 / 308 | 96 / 100 / 1440 |

Sensitivity: prevalence (recall % / precision % / contacts)

| prevalence | flagged | random same budget | exhaustive |
|---|---|---|---|
| 0.05 | 35 / 100 / 103 | 6 / 100 / 103 | 94 / 100 / 1440 |
| 0.1 | 51 / 100 / 172 | 11 / 100 / 172 | 94 / 100 / 1440 |
| 0.2 | 67 / 100 / 308 | 20 / 99 / 308 | 95 / 100 / 1440 |
| 0.4 | 80 / 100 / 582 | 39 / 100 / 582 | 95 / 100 / 1440 |

Sensitivity: vision_threshold (recall % / precision % / contacts)

| vision_threshold | flagged | random same budget | exhaustive |
|---|---|---|---|
| 0.2 | 68 / 100 / 326 | 21 / 100 / 326 | 95 / 100 / 1440 |
| 0.35 | 68 / 100 / 311 | 20 / 100 / 311 | 95 / 100 / 1440 |
| 0.5 | 67 / 100 / 308 | 20 / 99 / 308 | 95 / 100 / 1440 |
| 0.65 | 66 / 100 / 285 | 19 / 100 / 285 | 95 / 100 / 1440 |
| 0.8 | 65 / 100 / 283 | 18 / 100 / 283 | 95 / 100 / 1440 |

**Source table.** One faulty scout among eight.

| Configuration | Recall (%) | Precision (%) | F1 (%) |
|---|---|---|---|
| no_fault_single | 97.4 ± 1.4 | 90.5 ± 2.6 | 93.8 ± 1.6 |
| fault_single | 91.7 ± 4.7 | 72.7 ± 3.0 | 81.0 ± 2.3 |
| fault_double_average | 99.0 ± 1.2 | 97.5 ± 1.5 | 98.2 ± 0.9 |
| fault_double_crosscheck | 99.1 ± 1.1 | 97.2 ± 1.4 | 98.1 ± 0.9 |
| no_fault_double_average | 99.5 ± 0.7 | 99.6 ± 0.6 | 99.6 ± 0.5 |

diagnosis: {"faulty_scout_isolated": 1.0, "healthy_scouts_wrongly_isolated": 0.02} 

### swarm_nano32_r128_s0__Shear_Rot00

tier accuracy: {"vision_damage_recall": 0.948, "vision_false_alarm": 0.097, "ut_recall_single_contact": 0.601, "ut_false_alarm_single_contact": 0.02}

**Source table.** Tier 1 coverage of 480 zones against the number of scouts.

| Scouts | Time (min) | Sorties | Flight energy (Wh) | 5% loss: time (min) / coverage (%) | 10% loss: time / coverage | 20% loss: time / coverage |
|---|---|---|---|---|---|---|
| 1 | 31.9 | 6 | 3.54 | 29.1 / 91.5 | 25.8 / 81.3 | 20.0 / 63.8 |
| 2 | 15.1 | 6 | 3.43 | 16.3 / 100.0 | 17.9 / 99.6 | 19.1 / 95.1 |
| 4 | 8.9 | 8 | 3.91 | 9.4 / 100.0 | 10.1 / 100.0 | 12.1 / 100.0 |
| 8 | 4.0 | 8 | 3.92 | 4.4 / 100.0 | 5.0 / 100.0 | 5.6 / 100.0 |
| 12 | 2.9 | 12 | 4.35 | 3.1 / 100.0 | 3.6 / 100.0 | 4.3 / 100.0 |
| 16 | 2.4 | 16 | 4.65 | 2.7 / 100.0 | 3.0 / 100.0 | 3.6 / 100.0 |

**Source table.** Confirmation policies (mean ± standard deviation over simulated structures).

| Policy | Contacts | Tier 2 time (min) | Tier 2 energy (Wh) | Hidden-defect recall (%) | Precision (%) | Recall at threshold 0.2 (%) | Precision at threshold 0.2 (%) |
|---|---|---|---|---|---|---|---|
| Vision only | 0 | 0 | 0 | 0 ± 0 | nan ± nan | 69 | 23 |
| Random zones, same budget | 384 | 159 | 562 | 16 ± 6 | 100 ± 0 | 68 | 28 |
| Highest prior, half budget | 191 | 79 | 281 | 25 ± 6 | 100 ± 1 | 56 | 32 |
| Flagged zones | 384 | 159 | 561 | 44 ± 7 | 100 ± 0 | 48 | 100 |
| Flagged + 10% audit | 489 | 202 | 712 | 45 ± 7 | 100 ± 0 | 50 | 100 |
| Exhaustive | 1440 | 595 | 2088 | 61 ± 7 | 100 ± 1 | 66 | 100 |
| Perfect scout (oracle) | 288 | 119 | 419 | 44 ± 7 | 100 ± 1 | 51 | 38 |

n_hidden mean: 40.8  tier1 time: 4.0 min, tier1 energy: 3.92 Wh

Sensitivity: k_contacts (recall % / precision % / contacts)

| k_contacts | flagged | random same budget | exhaustive |
|---|---|---|---|
| 1 | 39 / 95 / 129 | 14 / 93 / 129 | 52 / 92 / 480 |
| 2 | 43 / 100 / 257 | 15 / 100 / 257 | 57 / 100 / 960 |
| 3 | 45 / 100 / 386 | 16 / 100 / 386 | 62 / 100 / 1440 |
| 5 | 50 / 100 / 644 | 20 / 100 / 644 | 68 / 100 / 2400 |

Sensitivity: ut_noise (recall % / precision % / contacts)

| ut_noise | flagged | random same budget | exhaustive |
|---|---|---|---|
| 0.0 | 45 / 100 / 386 | 16 / 100 / 386 | 62 / 100 / 1440 |
| 0.2 | 39 / 100 / 386 | 13 / 100 / 386 | 51 / 100 / 1440 |
| 0.4 | 35 / 100 / 386 | 11 / 100 / 386 | 43 / 100 / 1440 |
| 0.6 | 31 / 97 / 386 | 9 / 97 / 386 | 36 / 97 / 1440 |
| 0.8 | 28 / 54 / 386 | 8 / 49 / 386 | 32 / 51 / 1440 |
| 1.0 | 27 / 23 / 386 | 8 / 12 / 386 | 31 / 12 / 1440 |

Sensitivity: lift (recall % / precision % / contacts)

| lift | flagged | random same budget | exhaustive |
|---|---|---|---|
| 0.3_0.3 | 17 / 100 / 386 | 16 / 100 / 386 | 64 / 100 / 1440 |
| 0.3_0.1 | 30 / 100 / 386 | 16 / 100 / 386 | 62 / 100 / 1440 |
| 0.3_0.03 | 45 / 100 / 386 | 16 / 100 / 386 | 62 / 100 / 1440 |
| 0.3_0.01 | 55 / 100 / 386 | 16 / 100 / 386 | 62 / 100 / 1440 |
| 0.5_0.01 | 59 / 100 / 386 | 19 / 91 / 386 | 65 / 100 / 1440 |

Sensitivity: prevalence (recall % / precision % / contacts)

| prevalence | flagged | random same budget | exhaustive |
|---|---|---|---|
| 0.05 | 25 / 100 / 203 | 9 / 100 / 203 | 60 / 100 / 1440 |
| 0.1 | 35 / 100 / 264 | 10 / 99 / 264 | 61 / 100 / 1440 |
| 0.2 | 45 / 100 / 386 | 16 / 100 / 386 | 62 / 100 / 1440 |
| 0.4 | 53 / 100 / 632 | 27 / 100 / 632 | 62 / 100 / 1440 |

Sensitivity: vision_threshold (recall % / precision % / contacts)

| vision_threshold | flagged | random same budget | exhaustive |
|---|---|---|---|
| 0.2 | 49 / 100 / 568 | 24 / 100 / 568 | 62 / 100 / 1440 |
| 0.35 | 47 / 100 / 440 | 19 / 100 / 440 | 62 / 100 / 1440 |
| 0.5 | 45 / 100 / 386 | 16 / 100 / 386 | 62 / 100 / 1440 |
| 0.65 | 44 / 100 / 358 | 16 / 100 / 358 | 62 / 100 / 1440 |
| 0.8 | 40 / 100 / 313 | 14 / 100 / 313 | 62 / 100 / 1440 |

**Source table.** One faulty scout among eight.

| Configuration | Recall (%) | Precision (%) | F1 (%) |
|---|---|---|---|
| no_fault_single | 94.9 ± 2.5 | 71.2 ± 2.6 | 81.3 ± 1.9 |
| fault_single | 89.6 ± 4.6 | 60.3 ± 2.3 | 72.0 ± 2.4 |
| fault_double_average | 96.6 ± 1.9 | 71.6 ± 3.0 | 82.2 ± 2.2 |
| fault_double_crosscheck | 97.4 ± 1.7 | 74.9 ± 3.1 | 84.6 ± 2.2 |
| no_fault_double_average | 98.7 ± 1.2 | 76.3 ± 2.8 | 86.0 ± 1.9 |

diagnosis: {"faulty_scout_isolated": 0.99, "healthy_scouts_wrongly_isolated": 0.04} 

### swarm_nano32_r128_s0__Shear_Rot90

tier accuracy: {"vision_damage_recall": 0.948, "vision_false_alarm": 0.097, "ut_recall_single_contact": 0.874, "ut_false_alarm_single_contact": 0.018}

**Source table.** Tier 1 coverage of 480 zones against the number of scouts.

| Scouts | Time (min) | Sorties | Flight energy (Wh) | 5% loss: time (min) / coverage (%) | 10% loss: time / coverage | 20% loss: time / coverage |
|---|---|---|---|---|---|---|
| 1 | 31.9 | 6 | 3.54 | 28.3 / 89.1 | 25.2 / 79.5 | 19.9 / 63.5 |
| 2 | 15.1 | 6 | 3.43 | 16.6 / 99.7 | 18.1 / 99.2 | 18.7 / 93.6 |
| 4 | 8.9 | 8 | 3.91 | 9.7 / 100.0 | 10.4 / 100.0 | 12.0 / 99.8 |
| 8 | 4.0 | 8 | 3.92 | 4.5 / 100.0 | 5.0 / 100.0 | 5.7 / 100.0 |
| 12 | 2.9 | 12 | 4.35 | 3.2 / 100.0 | 3.7 / 100.0 | 4.3 / 100.0 |
| 16 | 2.4 | 16 | 4.65 | 2.7 / 100.0 | 3.1 / 100.0 | 3.6 / 100.0 |

**Source table.** Confirmation policies (mean ± standard deviation over simulated structures).

| Policy | Contacts | Tier 2 time (min) | Tier 2 energy (Wh) | Hidden-defect recall (%) | Precision (%) | Recall at threshold 0.2 (%) | Precision at threshold 0.2 (%) |
|---|---|---|---|---|---|---|---|
| Vision only | 0 | 0 | 0 | 0 ± 0 | nan ± nan | 69 | 22 |
| Random zones, same budget | 384 | 159 | 561 | 25 ± 6 | 100 ± 1 | 76 | 30 |
| Highest prior, half budget | 191 | 79 | 280 | 38 ± 7 | 100 ± 0 | 67 | 36 |
| Flagged zones | 384 | 159 | 560 | 68 ± 7 | 100 ± 1 | 68 | 100 |
| Flagged + 10% audit | 488 | 202 | 711 | 70 ± 7 | 100 ± 1 | 71 | 100 |
| Exhaustive | 1440 | 595 | 2088 | 94 ± 4 | 100 ± 1 | 96 | 100 |
| Perfect scout (oracle) | 288 | 119 | 419 | 68 ± 7 | 100 ± 0 | 72 | 46 |

n_hidden mean: 40.2  tier1 time: 4.0 min, tier1 energy: 3.92 Wh

Sensitivity: k_contacts (recall % / precision % / contacts)

| k_contacts | flagged | random same budget | exhaustive |
|---|---|---|---|
| 1 | 59 / 96 / 128 | 22 / 93 / 128 | 80 / 93 / 480 |
| 2 | 63 / 100 / 256 | 23 / 99 / 256 | 88 / 99 / 960 |
| 3 | 67 / 100 / 384 | 24 / 100 / 384 | 94 / 100 / 1440 |
| 5 | 70 / 100 / 641 | 28 / 100 / 641 | 98 / 100 / 2400 |

Sensitivity: ut_noise (recall % / precision % / contacts)

| ut_noise | flagged | random same budget | exhaustive |
|---|---|---|---|
| 0.0 | 67 / 100 / 384 | 24 / 100 / 384 | 94 / 100 / 1440 |
| 0.2 | 65 / 100 / 384 | 23 / 100 / 384 | 88 / 100 / 1440 |
| 0.4 | 61 / 99 / 384 | 21 / 99 / 384 | 78 / 99 / 1440 |
| 0.6 | 52 / 98 / 384 | 17 / 98 / 384 | 62 / 97 / 1440 |
| 0.8 | 39 / 60 / 384 | 12 / 58 / 384 | 45 / 58 / 1440 |
| 1.0 | 28 / 23 / 384 | 8 / 12 / 384 | 32 / 12 / 1440 |

Sensitivity: lift (recall % / precision % / contacts)

| lift | flagged | random same budget | exhaustive |
|---|---|---|---|
| 0.3_0.3 | 26 / 100 / 384 | 25 / 100 / 384 | 95 / 100 / 1440 |
| 0.3_0.1 | 44 / 100 / 384 | 25 / 100 / 384 | 94 / 100 / 1440 |
| 0.3_0.03 | 67 / 100 / 384 | 24 / 100 / 384 | 94 / 100 / 1440 |
| 0.3_0.01 | 82 / 100 / 384 | 25 / 100 / 384 | 95 / 100 / 1440 |
| 0.5_0.01 | 85 / 100 / 384 | 28 / 93 / 384 | 95 / 100 / 1440 |

Sensitivity: prevalence (recall % / precision % / contacts)

| prevalence | flagged | random same budget | exhaustive |
|---|---|---|---|
| 0.05 | 37 / 100 / 201 | 15 / 100 / 201 | 93 / 100 / 1440 |
| 0.1 | 52 / 100 / 262 | 17 / 99 / 262 | 94 / 100 / 1440 |
| 0.2 | 67 / 100 / 384 | 24 / 100 / 384 | 94 / 100 / 1440 |
| 0.4 | 80 / 100 / 630 | 41 / 100 / 630 | 95 / 100 / 1440 |

Sensitivity: vision_threshold (recall % / precision % / contacts)

| vision_threshold | flagged | random same budget | exhaustive |
|---|---|---|---|
| 0.2 | 74 / 100 / 568 | 38 / 100 / 568 | 94 / 100 / 1440 |
| 0.35 | 70 / 100 / 438 | 29 / 100 / 438 | 94 / 100 / 1440 |
| 0.5 | 67 / 100 / 384 | 24 / 100 / 384 | 94 / 100 / 1440 |
| 0.65 | 65 / 100 / 356 | 23 / 100 / 356 | 94 / 100 / 1440 |
| 0.8 | 60 / 100 / 308 | 20 / 100 / 308 | 94 / 100 / 1440 |

**Source table.** One faulty scout among eight.

| Configuration | Recall (%) | Precision (%) | F1 (%) |
|---|---|---|---|
| no_fault_single | 94.7 ± 2.4 | 71.1 ± 3.1 | 81.2 ± 2.2 |
| fault_single | 89.3 ± 4.3 | 60.4 ± 2.6 | 72.0 ± 2.4 |
| fault_double_average | 96.5 ± 1.8 | 71.4 ± 3.4 | 82.1 ± 2.4 |
| fault_double_crosscheck | 97.5 ± 1.6 | 74.5 ± 3.4 | 84.4 ± 2.4 |
| no_fault_double_average | 98.7 ± 1.2 | 76.0 ± 3.3 | 85.8 ± 2.2 |

diagnosis: {"faulty_scout_isolated": 0.99, "healthy_scouts_wrongly_isolated": 0.05} 

