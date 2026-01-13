# Current Status: FC vs SL Misclassification

## Analysis of Confusion Matrix
Based on the provided heatmap (`uploaded_image_1768275500957.png`), the most significant remaining misclassification cluster involves **Cutters (FC)** and **Sliders (SL)**.

### Error Counts
*   **FC (True) -> SL (Pred)**: **2,858 cases**
    *   This is the single largest off-diagonal error count in the matrix.
    *   Only 13,053 FC were correctly classified. The 2,858 errors represent a substantial portion of FC samples.
*   **SL (True) -> FC (Pred)**: **1,997 cases**
    *   This is the second largest source of confusion for Sliders.

### Total Confusion Volume
*   **Total FC <-> SL Errors**: 4,855 misclassified samples.

## Why this happens (Hypothesis)
Cutters (FC) and Sliders (SL) are biomechanically very similar.
*   **Velocity**: Cutters are usually faster (88-92 mph) than Sliders (80-88 mph), but there is overlap.
*   **Movement**: Both have horizontal break (glove-side), but Sliders typically break more.
*   **Ambiguity**: In the "grey zone", labeling can be inconsistent.

## Next Plans
1.  **Feature Analysis**: Compare the distributions of `release_speed` and `pfx_x` (horizontal break) for the misclassified samples.
2.  **Strategy**:
    *   **Feature Engineering**: Create features to separate them better (e.g., Break Ratio).
    *   **Filtering**: Identify if these are labeling errors.
3.  **Execution**: Create `analyze_fc_sl.ipynb` to investigate these cases.
