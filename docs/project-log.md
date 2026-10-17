## Wednesday September 23, 2026

### Goal
Build a working Wi-Fi CSI sensing pipeline using ESP32-C5 boards and train a baseline machine-learning classifier for human activity sensing.

### Hardware Setup
- 2× ESP32-C5 development boards
- 2× external dual-band Wi-Fi antennas
- Ubuntu laptop for flashing, data collection, and ML training

### ESP32 / CSI Setup
- Installed ESP-IDF.
- Initially used ESP-IDF 6.1.
- Successfully compiled the `csi_send` example after updating deprecated Wi-Fi bandwidth enum names:
  - `WIFI_BW_HT20` → `WIFI_BW20`
  - `WIFI_BW_HT40` → `WIFI_BW40`
- Encountered a receiver build failure because the `esp_csi_gain_ctrl` component was not available for the ESP-IDF 6.1 setup.
- Installed ESP-IDF 5.5.3 and rebuilt the receiver successfully.
- Flashed one ESP32-C5 as the transmitter and one as the receiver.

### First Successful CSI Link
- TX firmware successfully initialized on Wi-Fi channel 11.
- RX began receiving live `CSI_DATA` packets from TX.
- Verified that CSI traffic increased immediately when the transmitter was powered on.
- Captured raw CSI serial output to text files.

### CSI Visualization
- Parsed CSI arrays containing alternating I/Q values.
- Converted I/Q values into CSI amplitude.
- Generated a CSI heatmap showing changes across subcarriers over time.

### Dataset Collection
Collected 15 labeled trials across three classes:

- Empty room
- Person standing still
- Person walking

One incorrectly labeled still trial was moved into the walking class and replaced with a correctly recorded still trial.

### Machine Learning Baseline
- Parsed 29,996 valid CSI packets.
- 5 corrupted packets were skipped.
- CSI vectors contained 234 I/Q values per packet.
- Used 100-packet windows.
- Generated 294 usable windows.
- Used whole-trial train/test separation to reduce leakage.

Training set:
- 12 trials
- 238 windows

Test set:
- 3 held-out trials
- 56 windows

Model:
- Random Forest classifier

### Baseline Results
- Test accuracy: **96.43%**
- Empty: 22/22 test windows correct
- Still: 18/19 correct
- Walking: 14/15 correct

Confusion occurred only between the still and walking classes.

### Current Limitations
- Small dataset
- Single room
- Fixed transmitter and receiver positions
- One person used for data collection
- Only three activity classes
- Current model has not yet been tested in a different environment

### To-Do
- Test the existing model on completely new data without retraining
- Move TX/RX positions and test generalization
- Vary walking speed and direction
- Test additional people
- Add real-time inference
- Compare Random Forest against neural-network approaches
- Evaluate whether additional ESP32 nodes improve robustnessg into a truck load of issues
- get a better physical placement for the hardware


---

## October 2026

### Goal
Expand the original activity-sensing pipeline into a real-time 2D indoor localization system using Wi-Fi CSI.

### Localization Dataset
Created a 3×3 position grid with nine labeled positions:

- P1 = (-2 ft, 2 ft)
- P2 = (0 ft, 2 ft)
- P3 = (2 ft, 2 ft)
- P4 = (-2 ft, 4 ft)
- P5 = (0 ft, 4 ft)
- P6 = (2 ft, 4 ft)
- P7 = (-2 ft, 6 ft)
- P8 = (0 ft, 6 ft)
- P9 = (2 ft, 6 ft)

Collected five trials per position.

Each valid CSI packet contained:

- 234 alternating I/Q values
- 117 amplitude values after I/Q conversion

### Localization Training Pipeline
Created a dedicated localization training pipeline using:

- full `CSI_DATA` packet parsing
- 234 I/Q → 117 amplitude conversion
- 100-packet windows
- Random Forest classification
- trial-level train/test separation
- localization error calculations using physical grid coordinates
- saved model metadata to preserve preprocessing and feature compatibility

### Initial Localization Results
The first localization model produced:

- Test exact-position accuracy: **51.57%**
- Mean localization error: **1.311 ft**
- Within-2-ft accuracy: **75.47%**

Some positions were much more reliable than others, especially P4, P8, and P9.

P2 and P6 were among the weaker positions.

### Normalization Study
Created `evaluate_normalization.py` to compare several CSI preprocessing methods using complete-trial cross-validation.

Tested:

- raw amplitude
- per-packet mean normalization
- per-packet L2 normalization
- per-packet z-score normalization
- per-window normalization
- relative-shape normalization

Best result:

**relative_shape**

Results:

- Mean exact-position accuracy: **78.03%**
- Standard deviation: **8.27%**
- Mean localization error: **0.617 ft**
- Within-2-ft accuracy: **89.70%**

Raw amplitude baseline:

- Mean exact-position accuracy: **64.18%**
- Mean localization error: **0.905 ft**
- Within-2-ft accuracy: **87.04%**

The relative-shape method became the primary preprocessing method for later experiments.

### Relative-Shape Model
Created `train_localization_relative.py`.

The model:

- uses the same CSI parser
- uses 117 amplitude values
- uses 100-packet windows
- applies relative-shape preprocessing
- preserves exact feature ordering
- stores preprocessing details in model metadata

The real-time inference pipeline automatically reads the preprocessing method from metadata and rejects incompatible models.

### Real-Time Localization
Created `realtime_localization.py`.

Features include:

- live CSI input
- 100-packet windows
- overlapping windows
- default stride of 25 packets
- Random Forest `predict_proba`
- probability-weighted X/Y estimates
- EMA smoothing
- configurable confidence threshold
- uncertain-state handling
- CSV logging
- optional GUI
- packet diagnostics

At approximately 90 CSI packets per second, the overlapping-window system produces around **3.6 localization updates per second**.

### Real-Time Compatibility Fix
The original real-time script required the entire embedded model metadata object to exactly match `model_metadata.json`.

This was too strict.

Compatibility validation was changed so only inference-critical fields are checked, including:

- raw CSI I/Q length
- amplitude vector length
- window size
- exact feature names
- feature order
- feature count
- model `n_features_in_`
- class labels and class order
- coordinate mapping
- preprocessing parameters

Non-inference fields such as timestamps and absolute paths are ignored.

### UDP Wireless CSI Transport
The original receiver required USB serial between the ESP32-C5 receiver and the laptop.

The receiver firmware was modified to support wireless CSI transport over UDP.

New architecture:

TX ESP32-C5  
→ Wi-Fi CSI  
→ RX ESP32-C5  
→ UDP  
→ laptop

The receiver creates a Wi-Fi SoftAP called:

`CSI-RX`

The laptop connects directly to the receiver and listens on UDP port 5005.

The UDP protocol preserves:

- sequence number
- RSSI
- full compensated CSI array
- all 234 I/Q values

### UDP Validation
Wireless transport was validated successfully.

Observed performance:

- approximately **90.42 CSI packets/sec**
- 5,426 packets during a 60-second validation run
- full CSI arrays preserved
- RSSI matched serial output
- zero malformed packets during validation
- no duplicate or out-of-order packets
- only very small sequence gaps

The wireless transport itself appears stable enough for localization experiments.

### Wireless Recording
Updated `record_localization.py` to support:

- `--transport serial`
- `--transport udp`

UDP recording:

- uses the same existing UDP decoder
- reconstructs packets into the same `CSI_DATA` text format used by previous datasets
- preserves all 234 I/Q values
- counts malformed packets
- counts sequence gaps
- counts duplicates
- counts out-of-order packets
- supports the same output-directory structure as serial recording

This allows datasets to be collected without connecting the receiver to the laptop with USB.

### Geometry Experiment 2
Changed the physical TX/RX geometry and collected new trials 06 and 07 for all nine positions.

Relative-shape reciprocal evaluation:

Fold A:
- Train on trial 06
- Test on trial 07
- Exact accuracy: **24.53%**
- Mean error: **2.339 ft**
- Within-2-ft accuracy: **60.38%**

Fold B:
- Train on trial 07
- Test on trial 06
- Exact accuracy: **23.46%**
- Mean error: **2.587 ft**
- Within-2-ft accuracy: **40.74%**

Mean:

- Exact accuracy: **23.99%**
- Mean error: **2.463 ft**
- Within-2-ft accuracy: **50.56%**

These results were substantially worse than the earlier five-trial dataset.

However, this comparison was not fully equivalent because the earlier experiment trained on multiple trials per position, while these geometry tests trained on only one trial per position.

### 6 ft × 8 ft Diagonal Geometry
Created a new sensing area measuring approximately 6 ft × 8 ft.

Coordinates:

- P1 = (5, 1)
- P2 = (3, 1)
- P3 = (1, 1)
- P4 = (5, 4)
- P5 = (3, 4)
- P6 = (1, 4)
- P7 = (5, 7)
- P8 = (3, 7)
- P9 = (1, 7)

A new diagonal TX/RX layout was tested.

Two wireless trials were collected at every position.

### Diagonal Geometry Results
Using relative-shape preprocessing:

Fold A:
- Train on trial 01
- Test on trial 02
- Exact accuracy: **47.71%**
- Mean error: **1.610 ft**
- Within-2-ft accuracy: **64.05%**

Fold B:
- Train on trial 02
- Test on trial 01
- Exact accuracy: **42.58%**
- Mean error: **1.966 ft**
- Within-2-ft accuracy: **50.97%**

Mean:

- Exact accuracy: **45.15%**
- Mean localization error: **1.788 ft**
- Within-2-ft accuracy: **57.51%**

Per-position performance varied significantly.

Strong positions included:

- P3
- P6

Weak positions included:

- P2
- P5
- P8

### Diagonal Normalization Comparison
The same diagonal dataset was evaluated using multiple preprocessing methods.

The best average exact-position accuracy was approximately:

**49.34% using packet z-score normalization**

This suggested that preprocessing alone was not enough to solve the remaining localization error.

### Current Interpretation
The experiments so far suggest that localization performance depends strongly on:

- TX/RX physical placement
- antenna orientation
- environmental stability
- number of independent training trials
- preprocessing method
- trial-to-trial variation

An important discovery was that the earlier **78.03%** result and later **23–49%** results were not directly comparable.

The earlier experiment used multiple training trials per position.

The later geometry experiments used only one training trial per position in each fold.

This means part of the performance difference may be caused by dataset size rather than geometry alone.

### Planned Hardware Improvement
The next hardware design will separate CSI sensing from Wi-Fi networking.

Current RX ESP32-C5:

CSI reception  
+ SoftAP  
+ UDP transmission

Planned architecture:

TX ESP32-C5  
→ CSI  
→ dedicated sensing RX ESP32-C5  
→ UART  
→ second ESP32 bridge  
→ UDP  
→ laptop

The sensing receiver will focus only on CSI acquisition.

The second ESP32 will handle networking and UDP transport.

The purpose is to make the sensing path more controlled and reduce possible interaction between CSI acquisition and networking activity.

### Planned Large Dataset
The next dataset will use a fixed geometry and a much larger number of independent trials.

Planned collection:

- 9 positions
- 36 trials per position
- 30 seconds per trial
- 324 total recordings
- approximately 2,700 packets per trial at ~90 packets/sec
- multiple recording sessions across different days/times

Planned structure:

- 6 sessions
- 6 trials per position per session
- randomized position order
- complete physical repositioning between trials

Planned evaluation:

- Sessions 1–4: training
- Session 5: validation
- Session 6: final held-out testing

The purpose of this experiment is to determine whether the current localization system is mainly limited by insufficient training diversity or by the information available from a single Wi-Fi CSI sensing link.

### Current Next Steps
- Separate sensing and UDP networking onto different ESP32 boards
- Freeze a new physical geometry
- Collect the 324-recording dataset
- Compare preprocessing methods again
- Evaluate held-out session generalization
- Determine whether a second CSI link is necessary
