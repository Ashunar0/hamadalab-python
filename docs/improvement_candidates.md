# Improvement Candidates (Based on Confusion Matrix)

Besides **FC vs SL**, here are the other significant misclassification clusters identified:

## 1. Slider (SL) vs Sweeper (ST)
*   **Total Errors**: **2,622**
*   **Breakdown**:
    *   SL (True) -> ST (Pred): **1,411**
    *   ST (True) -> SL (Pred): **1,211**
*   **Context**: "Sweeper" is a relatively new classification (a type of slider with large horizontal break). The boundary between a "normal" slider and a "sweeper" is often fluid, leading to high confusion.

## 2. Sinker (SI) vs 4-Seam Fastball (FF)
*   **Total Errors**: **2,012**
*   **Breakdown**:
    *   SI (True) -> FF (Pred): **1,479**
    *   FF (True) -> SI (Pred): **533**
*   **Context**: Sinkers are fastballs with arm-side run and sink. If the sink/run isn't pronounced enough, the model (and sometimes scouts) confuses them with straight 4-seamers. The high number of SI->FF suggests many sinkers are being classified as generic fastballs.

## 3. Curveball (CU) vs Knuckle Curve (KC)
*   **Total Errors**: **1,033**
*   **Breakdown**:
    *   KC (True) -> CU (Pred): **725**
    *   CU (True) -> KC (Pred): **308**
*   **Context**: Biomechanically different grip, but trajectory can be very similar.

## 4. Changeup (CH) vs Sinker (SI)
*   **Total Errors**: **1,071**
*   **Breakdown**:
    *   CH (True) -> SI (Pred): **612**
    *   SI (True) -> CH (Pred): **459**
*   **Context**: Both have arm-side fade. Hard changeups can look like slow sinkers.

---
**Recommendation**: 
After **FC/SL (4,855 errors)**, the **SL/ST (2,622 errors)** pair is the next largest target for improvement.
