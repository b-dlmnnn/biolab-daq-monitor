# 🔬 BioLab-DAQ: Microfluidic & Optical Biosensor Experiment Monitor

Real-time data acquisition (DAQ) and digital signal processing (DSP) dashboard designed for microfluidic thermal-optical biosensor benches.

## Key Modules
1. **Real-Time DAQ & DSP Filtering:** Applies median despiking and exponential moving average (EMA) filtering on noisy thermistor and photodiode streams while computing Signal-to-Noise Ratio (SNR in dB).
2. **Standard Calibration Curve ($R^2$ & LOD):** Performs OLS linear regression across triplicate reference standards and calculates the $3\sigma$ Limit of Detection (LOD).
3. **Electronic Lab Notebook (SQLite):** Logs experimental runs with chip IDs, assay protocols, and CSV export support.
