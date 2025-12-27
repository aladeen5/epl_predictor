# database.py - Database utilities for match storage and retrieval

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os


class MatchDatabase:
    def __init__(self, db_path='epl_matches.db'):
        self.db_path = db_path
        self.conn = None
        self.initialize_database()

    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def initialize_database(self):
        """Create tables if they don't exist"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Matches table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS matches
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           date
                           TEXT
                           NOT
                           NULL,
                           season
                           TEXT
                           NOT
                           NULL,
                           home_team
                           TEXT
                           NOT
                           NULL,
                           away_team
                           TEXT
                           NOT
                           NULL,
                           home_goals
                           INTEGER
                           NOT
                           NULL,
                           away_goals
                           INTEGER
                           NOT
                           NULL,
                           result
                           TEXT
                           NOT
                           NULL,
                           home_shots
                           INTEGER
                           DEFAULT
                           0,
                           away_shots
                           INTEGER
                           DEFAULT
                           0,
                           home_shots_target
                           INTEGER
                           DEFAULT
                           0,
                           away_shots_target
                           INTEGER
                           DEFAULT
                           0,
                           home_corners
                           INTEGER
                           DEFAULT
                           0,
                           away_corners
                           INTEGER
                           DEFAULT
                           0,
                           home_fouls
                           INTEGER
                           DEFAULT
                           0,
                           away_fouls
                           INTEGER
                           DEFAULT
                           0,
                           home_yellows
                           INTEGER
                           DEFAULT
                           0,
                           away_yellows
                           INTEGER
                           DEFAULT
                           0,
                           home_reds
                           INTEGER
                           DEFAULT
                           0,
                           away_reds
                           INTEGER
                           DEFAULT
                           0,
                           created_at
                           TEXT
                           DEFAULT
                           CURRENT_TIMESTAMP
                       )
                       ''')

        # Predictions table (for logging)
        # In database.py, update the predictions table creation:

        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS predictions
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           prediction_date
                           TEXT
                           NOT
                           NULL,
                           match_date
                           TEXT
                           NOT
                           NULL,
                           home_team
                           TEXT
                           NOT
                           NULL,
                           away_team
                           TEXT
                           NOT
                           NULL,
                           predicted_result
                           TEXT
                           NOT
                           NULL,
                           prob_home
                           REAL
                           NOT
                           NULL,
                           prob_draw
                           REAL
                           NOT
                           NULL,
                           prob_away
                           REAL
                           NOT
                           NULL,
                           confidence
                           REAL
                           NOT
                           NULL,
                           predicted_home_goals
                           INTEGER
                           DEFAULT
                           NULL,
                           predicted_away_goals
                           INTEGER
                           DEFAULT
                           NULL,
                           actual_home_goals
                           INTEGER
                           DEFAULT
                           NULL,
                           actual_away_goals
                           INTEGER
                           DEFAULT
                           NULL,
                           actual_result
                           TEXT
                           DEFAULT
                           NULL,
                           correct
                           INTEGER
                           DEFAULT
                           NULL,
                           created_at
                           TEXT
                           DEFAULT
                           CURRENT_TIMESTAMP
                       )
                       ''')
        conn.commit()
        conn.close()
        print("✓ Database initialized")

    def load_historical_data(self, df):
        """Load historical data from DataFrame into database"""
        conn = self.get_connection()

        # Prepare data for insertion
        df_insert = df.copy()

        # Map column names to database schema
        column_mapping = {
            'Date': 'date',
            'Season': 'season',
            'HomeTeam': 'home_team',
            'AwayTeam': 'away_team',
            'FTHG': 'home_goals',
            'FTAG': 'away_goals',
            'FTR': 'result',
            'HS': 'home_shots',
            'AS': 'away_shots',
            'HST': 'home_shots_target',
            'AST': 'away_shots_target',
            'HC': 'home_corners',
            'AC': 'away_corners',
            'HF': 'home_fouls',
            'AF': 'away_fouls',
            'HY': 'home_yellows',
            'AY': 'away_yellows',
            'HR': 'home_reds',
            'AR': 'away_reds',
        }

        # Select and rename columns
        available_cols = [col for col in column_mapping.keys() if col in df_insert.columns]
        df_insert = df_insert[available_cols].copy()
        df_insert = df_insert.rename(columns=column_mapping)

        # Convert date to string format
        if df_insert['date'].dtype != 'object':
            df_insert['date'] = pd.to_datetime(df_insert['date']).dt.strftime('%Y-%m-%d')

        # Fill NaN values
        numeric_cols = ['home_shots', 'away_shots', 'home_shots_target', 'away_shots_target',
                        'home_corners', 'away_corners', 'home_fouls', 'away_fouls',
                        'home_yellows', 'away_yellows', 'home_reds', 'away_reds']

        for col in numeric_cols:
            if col in df_insert.columns:
                df_insert[col] = df_insert[col].fillna(0).astype(int)

        # Insert into database
        df_insert.to_sql('matches', conn, if_exists='append', index=False)

        conn.close()
        print(f"✓ Loaded {len(df_insert)} historical matches into database")

    def add_match(self, match_data):
        """Add a single match to database"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
                       INSERT INTO matches (date, season, home_team, away_team, home_goals, away_goals, result,
                                            home_shots, away_shots, home_shots_target, away_shots_target,
                                            home_corners, away_corners, home_fouls, away_fouls,
                                            home_yellows, away_yellows, home_reds, away_reds)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ''', (
                           match_data['date'],
                           match_data['season'],
                           match_data['home_team'],
                           match_data['away_team'],
                           match_data['home_goals'],
                           match_data['away_goals'],
                           match_data['result'],
                           match_data.get('home_shots', 0),
                           match_data.get('away_shots', 0),
                           match_data.get('home_shots_target', 0),
                           match_data.get('away_shots_target', 0),
                           match_data.get('home_corners', 0),
                           match_data.get('away_corners', 0),
                           match_data.get('home_fouls', 0),
                           match_data.get('away_fouls', 0),
                           match_data.get('home_yellows', 0),
                           match_data.get('away_yellows', 0),
                           match_data.get('home_reds', 0),
                           match_data.get('away_reds', 0),
                       ))

        conn.commit()
        conn.close()

    def get_all_matches(self):
        """Get all matches as DataFrame"""
        conn = self.get_connection()
        df = pd.read_sql_query("SELECT * FROM matches ORDER BY date", conn)
        conn.close()

        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])

        return df

    def get_team_matches(self, team, before_date=None):
        """Get all matches for a specific team"""
        conn = self.get_connection()

        if before_date:
            query = '''
                    SELECT * \
                    FROM matches
                    WHERE (home_team = ? OR away_team = ?) AND date < ?
                    ORDER BY date DESC \
                    '''
            df = pd.read_sql_query(query, conn, params=(team, team, before_date))
        else:
            query = '''
                    SELECT * \
                    FROM matches
                    WHERE home_team = ? \
                       OR away_team = ?
                    ORDER BY date DESC \
                    '''
            df = pd.read_sql_query(query, conn, params=(team, team))

        conn.close()

        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])

        return df

    def get_h2h_matches(self, team1, team2, before_date=None, limit=5):
        """Get head-to-head matches between two teams"""
        conn = self.get_connection()

        if before_date:
            query = '''
                    SELECT * \
                    FROM matches
                    WHERE ((home_team = ? AND away_team = ?) OR (home_team = ? AND away_team = ?))
                      AND date \
                        < ?
                    ORDER BY date DESC
                        LIMIT ? \
                    '''
            df = pd.read_sql_query(query, conn, params=(team1, team2, team2, team1, before_date, limit))
        else:
            query = '''
                    SELECT * \
                    FROM matches
                    WHERE (home_team = ? AND away_team = ?) \
                       OR (home_team = ? AND away_team = ?)
                    ORDER BY date DESC
                        LIMIT ? \
                    '''
            df = pd.read_sql_query(query, conn, params=(team1, team2, team2, team1, limit))

        conn.close()

        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])

        return df

    def get_all_teams(self):
        """Get list of all unique teams"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
                       SELECT DISTINCT home_team
                       FROM matches
                       UNION
                       SELECT DISTINCT away_team
                       FROM matches
                       ORDER BY home_team
                       ''')

        teams = [row[0] for row in cursor.fetchall()]
        conn.close()

        return teams

    def save_prediction(self, prediction_data):
        """Save a prediction to the database"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
                       INSERT INTO predictions (prediction_date, match_date, home_team, away_team,
                                                predicted_result, prob_home, prob_draw, prob_away, confidence,
                                                predicted_home_goals, predicted_away_goals)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ''', (
                           datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                           prediction_data['match_date'],
                           prediction_data['home_team'],
                           prediction_data['away_team'],
                           prediction_data['predicted_result'],
                           prediction_data['prob_home'],
                           prediction_data['prob_draw'],
                           prediction_data['prob_away'],
                           prediction_data['confidence'],
                           prediction_data.get('predicted_home_goals'),
                           prediction_data.get('predicted_away_goals')
                       ))

        conn.commit()
        conn.close()

        cursor.execute('''
                       INSERT INTO predictions (prediction_date, match_date, home_team, away_team,
                                                predicted_result, prob_home, prob_draw, prob_away, confidence)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ''', (
                           datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                           prediction_data['match_date'],
                           prediction_data['home_team'],
                           prediction_data['away_team'],
                           prediction_data['predicted_result'],
                           prediction_data['prob_home'],
                           prediction_data['prob_draw'],
                           prediction_data['prob_away'],
                           prediction_data['confidence'],
                       ))

        conn.commit()
        conn.close()

    def get_predictions(self, limit=None):
        """Get all predictions"""
        conn = self.get_connection()

        if limit:
            query = f"SELECT * FROM predictions ORDER BY prediction_date DESC LIMIT {limit}"
        else:
            query = "SELECT * FROM predictions ORDER BY prediction_date DESC"

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df

    def update_prediction_result(self, prediction_id, actual_result):
        """Update prediction with actual result"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Get the prediction
        cursor.execute("SELECT predicted_result FROM predictions WHERE id = ?", (prediction_id,))
        row = cursor.fetchone()

        if row:
            predicted = row[0]
            correct = 1 if predicted == actual_result else 0

            cursor.execute('''
                           UPDATE predictions
                           SET actual_result = ?,
                               correct       = ?
                           WHERE id = ?
                           ''', (actual_result, correct, prediction_id))

            conn.commit()

        conn.close()

    def get_prediction_accuracy(self):
        """Calculate overall prediction accuracy"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
                       SELECT COUNT(*)                                                      as total,
                              SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END)                  as correct,
                              AVG(CASE WHEN correct IS NOT NULL THEN correct ELSE NULL END) as accuracy
                       FROM predictions
                       WHERE actual_result IS NOT NULL
                       ''')

        result = cursor.fetchone()
        conn.close()

        return {
            'total': result[0],
            'correct': result[1] or 0,
            'accuracy': result[2] or 0.0
        }


# Test the database
if __name__ == "__main__":
    db = MatchDatabase()
    print("Database created successfully!")