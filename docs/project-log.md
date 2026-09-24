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
