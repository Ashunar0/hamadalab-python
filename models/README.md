# Pitch Classification Model Artifacts

This directory contains the finalized pitch classification model and its dependencies.

## Files
- `pitch_classifier_xgb.pkl`: The trained XGBoost model (Full Feature version, 95.19% accuracy).
- `label_encoder.pkl`: LabelEncoder for mapping numeric predictions to pitch type strings (FF, SL, etc.).
- `pitcher_stats.csv`: Average statistics for each pitcher, used to calculate pitcher-relative features.
- `global_stats.json`: Global average statistics used as a fallback for unknown pitchers.

## Performance Metrics (on Test Set)
- **Accuracy**: 0.9519
- **Weighted F1**: 0.9518
- **FC Recall**: 0.8830
- **SI Recall**: 0.9605

## Usage
Use the provided `predict.py` script located in the root directory.

```python
from predict import PitchPredictor

predictor = PitchPredictor(model_dir='models')
sample_data = [{
    'pitcher': 660271,
    'release_speed': 97.5,
    'release_spin_rate': 2400,
    'spin_axis': 200,
    'pfx_x': -0.5,
    'pfx_z': 1.5,
    'release_pos_x': -2.1,
    'release_pos_z': 5.9
}]
prediction = predictor.predict(sample_data)
print(prediction[0]['pitch_type']) # e.g., 'FF'
```
