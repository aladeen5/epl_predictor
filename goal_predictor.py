# goal_predictor.py - Predict exact match scores

import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error


class GoalPredictor:
    """
    Predicts exact number of goals for home and away teams
    Uses Poisson regression approach
    """

    def __init__(self):
        self.home_model = None
        self.away_model = None
        self.feature_columns = None

    def train(self, X_train, y_home_goals, y_away_goals, X_val, y_val_home, y_val_away):
        """
        Train separate models for home and away goals
        """
        print("\n" + "=" * 60)
        print("TRAINING GOAL PREDICTION MODELS")
        print("=" * 60)

        # Home goals model
        print("\nTraining home goals predictor...")
        self.home_model = xgb.XGBRegressor(
            objective='count:poisson',  # Poisson for count data
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            seed=42
        )

        self.home_model.fit(
            X_train, y_home_goals,
            eval_set=[(X_val, y_val_home)],
            verbose=False
        )

        # Away goals model
        print("Training away goals predictor...")
        self.away_model = xgb.XGBRegressor(
            objective='count:poisson',
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            seed=42
        )

        self.away_model.fit(
            X_train, y_away_goals,
            eval_set=[(X_val, y_val_away)],
            verbose=False
        )

        # Evaluate
        train_home_pred = self.home_model.predict(X_train)
        train_away_pred = self.away_model.predict(X_train)

        val_home_pred = self.home_model.predict(X_val)
        val_away_pred = self.away_model.predict(X_val)

        train_mae_home = mean_absolute_error(y_home_goals, train_home_pred)
        train_mae_away = mean_absolute_error(y_away_goals, train_away_pred)

        val_mae_home = mean_absolute_error(y_val_home, val_home_pred)
        val_mae_away = mean_absolute_error(y_val_away, val_away_pred)

        print(f"\nHome Goals MAE:")
        print(f"  Training: {train_mae_home:.3f}")
        print(f"  Validation: {val_mae_home:.3f}")

        print(f"\nAway Goals MAE:")
        print(f"  Training: {train_mae_away:.3f}")
        print(f"  Validation: {val_mae_away:.3f}")

        print("\n✓ Goal prediction models trained")

        return val_mae_home, val_mae_away

    def predict(self, X):
        """
        Predict goals for matches
        Returns: home_goals, away_goals (as floats)
        """
        home_goals = self.home_model.predict(X)
        away_goals = self.away_model.predict(X)

        # Clip to reasonable range
        home_goals = np.clip(home_goals, 0, 10)
        away_goals = np.clip(away_goals, 0, 10)

        return home_goals, away_goals

    def predict_score(self, X):
        """
        Predict most likely scoreline with probabilities
        Returns: list of dicts with prediction details
        """
        home_goals_mean, away_goals_mean = self.predict(X)

        results = []

        for i in range(len(X)):
            home_mean = home_goals_mean[i]
            away_mean = away_goals_mean[i]

            # Generate probability distribution for scores 0-5
            from scipy.stats import poisson

            score_probs = []
            for h in range(6):
                for a in range(6):
                    prob_h = poisson.pmf(h, home_mean)
                    prob_a = poisson.pmf(a, away_mean)
                    prob = prob_h * prob_a
                    score_probs.append((h, a, prob))

            # Sort by probability
            score_probs.sort(key=lambda x: x[2], reverse=True)

            results.append({
                'predicted_home_goals': round(home_mean),
                'predicted_away_goals': round(away_mean),
                'most_likely_scores': score_probs[:5],  # Top 5 most likely
                'home_expected': home_mean,
                'away_expected': away_mean
            })

        return results

    def save(self, filepath='goal_predictor_model.pkl'):
        """Save model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'home_model': self.home_model,
                'away_model': self.away_model,
                'feature_columns': self.feature_columns
            }, f)
        print(f"✓ Goal predictor saved to {filepath}")

    @classmethod
    def load(cls, filepath='goal_predictor_model.pkl'):
        """Load model"""
        predictor = cls()
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            predictor.home_model = data['home_model']
            predictor.away_model = data['away_model']
            predictor.feature_columns = data.get('feature_columns')
        print(f"✓ Goal predictor loaded from {filepath}")
        return predictor