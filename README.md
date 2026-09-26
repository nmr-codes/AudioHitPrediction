# Audio Hit Prediction & Gradient Boosting Benchmark
### Comparative Evaluation: LightGBM vs XGBoost with Optuna Hyperparameter Optimization

A production-grade machine learning project predicting commercial track popularity using acoustic profiles from 32,828 Spotify songs across 6 major genres.

---

## Performance Benchmark

All models evaluated on an identical stratified split (80% Train / 20% Test) across 14 tabular features:

| Estimator | Train Accuracy | Test Accuracy | Train ROC-AUC | Test ROC-AUC | Generalization Gap | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline LightGBM** (`num_leaves=31`) | 0.7497 | 0.6776 | 0.8300 | 0.7385 | +0.0721 | **0.240 s** |
| **Overfit-Controlled LightGBM** (`depth=7`, `min_child=20`) | 0.7423 | 0.6777 | 0.8175 | 0.7395 | **+0.0646** | 0.220 s |
| **XGBoost (Hist Tree Method)** (`n_estimators=100`) | 0.8615 | 0.6934 | 0.9380 | 0.7523 | +0.1681 | 0.928 s |
| **Optuna-Tuned LightGBM** (30-trial Bayesian) | 0.9229 | **0.7013** | 0.9788 | **0.7699** | +0.2216 | 0.445 s |

---

## Key Experimental Findings

1. **Model Accuracy**: Optuna-tuned LightGBM achieved the highest discriminative power with **0.770 ROC-AUC** and **70.13% test accuracy** on out-of-fold validation tracks.
2. **Computational Speed**: LightGBM outperformed XGBoost with a **3.8x faster training throughput** (0.240s vs 0.928s), confirming the efficiency of histogram binning and leaf-wise tree growth on tabular datasets.
3. **Overfitting Inflection**: As `num_leaves` increases past 31, variance inflation starts. At 128 to 512 leaves, the estimator collapsed into memorization (Train Acc: 99.3%, Test Acc: 71.4%, 27.8% gap).
4. **Regularization Control**: Constraining tree depth (`max_depth=7`) paired with leaf child sample limits (`min_child_samples=20`) and L2 penalty (`reg_lambda=3.27`) stabilized generalization performance.
5. **Feature Dominance**: Tree split distributions prove that `release_year`, `loudness`, `danceability`, and `duration_min` represent over 52% of all branching decisions.

---

## Project Structure

```text
├── app.py                      # Interactive Streamlit Web Application
├── main.py                     # Entry point launcher script
├── practise.ipynb              # Executed Jupyter Notebook with charts and analysis
├── pipeline_experiments.py     # End-to-end experiment pipeline & evaluation script
├── build_notebook.py           # Automated Jupyter notebook builder
├── requirements.txt            # Pinned dependencies
├── results.json                # Complete quantitative benchmark records
├── assets/                     # High-resolution benchmark figures
│   ├── num_leaves_overfitting.png
│   ├── overfitting_control_heatmap.png
│   ├── lgbm_vs_xgboost.png
│   ├── optuna_optimization_history.png
│   └── feature_importances.png
└── models/                     # Serialized trained model artifacts
    ├── lgbm_best.joblib
    ├── xgb_model.joblib
    └── pipeline_meta.joblib
```

---

## Quickstart

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone git@github.com:nmr-codes/AudioHitPrediction.git
cd AudioHitPrediction
pip install -r requirements.txt
```

### 2. Launch the Web Application
Start the interactive Streamlit dashboard:
```bash
streamlit run app.py
# or
python main.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Run Experiments from Scratch
To retrain all estimators, execute Optuna optimization, and regenerate visual artifacts:
```bash
python pipeline_experiments.py
```

### 4. Interactive Jupyter Notebook
Open `practise.ipynb` in VS Code or Jupyter to inspect pre-computed execution outputs, metric tables, and markdown commentary.

---

## License
MIT License.
