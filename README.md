# Premier League Match Predictor 🏆⚽

An XGBoost-based machine learning model for predicting English Premier League match outcomes with 57% accuracy.

## 📊 Model Performance

- **Test Accuracy**: 57.04%
- **Test Log Loss**: 0.9866
- **Training Data**: 2013/14 - 2024/25 seasons
- **Model Type**: XGBoost Multi-class Classification
- **Predictions**: Home Win (H), Draw (D), Away Win (A)

## 🎯 Key Features

The model uses 40+ features including:
- **Team Form**: Last 5 matches statistics (points, goals, wins)
- **Season Performance**: Overall goals scored/conceded, win rates, PPG
- **Head-to-Head**: Historical performance between specific teams
- **Comparative Metrics**: Form difference, goal-scoring difference, etc.
- **Temporal Features**: Month, day of week, match number

### Top 5 Most Important Features
1. Team form indicators (points in last 5 matches)
2. Goals scored/conceded averages
3. Head-to-head statistics
4. Win rate differentials
5. Points per game trends

## 📁 Project Structure

```
epl-predictor/
│
├── data_acquisition.ipynb          # Data collection and processing
├── epl_xgboost_model.json         # Trained XGBoost model
├── model_artifacts.pkl             # Label encoder, feature columns, etc.
├── model_info.json                 # Model metadata and parameters
│
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Load the Model

```python
import xgboost as xgb
import pickle
import pandas as pd
import numpy as np

# Load model
model = xgb.Booster()
model.load_model('epl_xgboost_model.json')

# Load artifacts
with open('model_artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)

label_encoder = artifacts['label_encoder']
feature_columns = artifacts['feature_columns']
```

### 2. Make Predictions

```python
# Prepare your match data with required features
match_features = prepare_match_features(home_team, away_team, historical_data)

# Create DMatrix
dmatrix = xgb.DMatrix(match_features[feature_columns])

# Predict
predictions = model.predict(dmatrix)
predicted_class = label_encoder.inverse_transform([np.argmax(predictions[0])])[0]

print(f"Predicted Result: {predicted_class}")
print(f"Probabilities - H: {predictions[0][2]:.2%}, D: {predictions[0][1]:.2%}, A: {predictions[0][0]:.2%}")
```

## 📊 Understanding Predictions

The model outputs three probabilities:
- **Index 0**: Away Win probability
- **Index 1**: Draw probability  
- **Index 2**: Home Win probability

**Confidence Levels**:
- High Confidence: >60% probability
- Medium Confidence: 50-60% probability
- Low Confidence: 40-50% probability
- Very Low: <40% probability

**Accuracy by Confidence**:
- Predictions with >60% confidence achieve ~65% accuracy
- Overall test set accuracy: 57.04%

## 🔧 Requirements

```
python>=3.8
xgboost>=2.0.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
```

Install dependencies:
```bash
pip install xgboost pandas numpy scikit-learn
```

## 📈 Model Details

**Algorithm**: XGBoost (Extreme Gradient Boosting)

**Best Hyperparameters**:
- Learning Rate: 0.1
- Max Depth: 5
- Number of Estimators: 300
- Subsample: 0.8
- Colsample by Tree: 0.8
- Min Child Weight: 3
- Gamma: 0.1
- Reg Alpha: 0.01
- Reg Lambda: 1.5

**Training Details**:
- Objective: Multi-class softprob
- Evaluation Metrics: Log Loss, Classification Error
- Early Stopping: 20 rounds
- Cross-Validation: 3-fold time-series split

## 🎲 Use Cases

1. **Match Outcome Prediction**: Predict H/D/A for upcoming fixtures
2. **Betting Analysis**: Compare model odds with bookmaker odds
3. **Team Performance Analysis**: Understand key performance indicators
4. **Fantasy Football**: Identify likely winners for captain selection
5. **Sports Analytics**: Research team strengths and patterns

## ⚠️ Important Notes

### Model Limitations
- **Football is unpredictable**: Even the best models cap at ~60% accuracy
- **Form can change rapidly**: Recent injuries, transfers, or manager changes affect performance
- **No match-specific context**: Model doesn't know about injuries, suspensions, or motivation
- **Historical bias**: Based on past patterns, unusual situations may not be captured

### Best Practices
1. **Don't use in isolation**: Combine with domain knowledge and current news
2. **Monitor confidence levels**: Higher confidence predictions are more reliable
3. **Update regularly**: Retrain model as new season data becomes available
4. **Consider context**: Use model as one input among many for decisions

## 🔄 Updating the Model

To retrain with new data:

1. Download latest season data from [football-data.co.uk](https://www.football-data.co.uk)
2. Append to existing dataset
3. Rerun feature engineering pipeline
4. Retrain XGBoost model with same hyperparameters
5. Evaluate on new test set (most recent season)

Recommended retraining frequency: **Every 3-4 months** during season

## 📊 Feature Engineering Pipeline

The model requires these calculated features:

**For each team** (Home and Away):
- Matches played
- Points in last 5 matches
- Goals scored in last 5 matches
- Goals conceded in last 5 matches
- Wins in last 5 matches
- Average goals scored (season)
- Average goals conceded (season)
- Win rate (season)
- Points per game (season)

**Head-to-Head**:
- Number of previous H2H matches
- Home wins in H2H
- Draws in H2H
- Away wins in H2H
- Average goals in H2H

**Comparative**:
- Form difference (home - away)
- Goal scoring difference
- Defense difference
- Win rate difference
- PPG difference

**Temporal**:
- Month (1-12)
- Day of week (0-6)
- Match number in season

## 📚 Data Source

Historical data sourced from [football-data.co.uk](https://www.football-data.co.uk/englandm.php)
- Coverage: 2013/14 to 2024/25 seasons
- Includes: Results, goals, shots, fouls, cards, corners
- Format: CSV files by season

## 🤝 Contributing

Improvements welcome! Areas for enhancement:
- Additional features (xG, possession, player-level stats)
- Ensemble methods (combine multiple models)
- Deep learning approaches (LSTM for time-series)
- Real-time data integration
- Web interface for predictions

## 📄 License

This project is for educational and research purposes. 

**Disclaimer**: This model is for informational purposes only. Do not use for gambling or financial decisions. Past performance does not guarantee future results.

## 🙏 Acknowledgments

- Football-Data.co.uk for historical match data
- XGBoost development team
- scikit-learn community

## 📧 Contact

For questions or suggestions about the model, please open an issue or submit a pull request.

---

**Last Updated**: December 2024  
**Model Version**: 1.0  
**Training Date Range**: August 2013 - December 2024