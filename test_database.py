# test_database.py - Load historical data into database

from database import MatchDatabase
import pandas as pd
import pickle

print("=" * 60)
print("LOADING HISTORICAL DATA INTO DATABASE")
print("=" * 60)

# Load the cleaned data from your notebook
# Option 1: If you saved df to CSV in your notebook
try:
    df = pd.read_csv('data/epl_cleaned.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    print("✓ Loaded data from CSV")
except:
    print("❌ Could not find 'cleaned_epl_data.csv'")
    print("Please run this in your notebook first:")
    print("    df.to_csv('cleaned_epl_data.csv', index=False)")
    exit()

print(f"Loaded {len(df)} matches")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# Initialize database
print("\nInitializing database...")
db = MatchDatabase()

# Check if database is already populated
existing_matches = db.get_all_matches()
if len(existing_matches) > 0:
    print(f"\n⚠️  Database already contains {len(existing_matches)} matches")
    response = input("Do you want to reload all data? (yes/no): ")
    if response.lower() != 'yes':
        print("Skipping data load...")
        exit()
    else:
        # Clear existing data
        import os
        os.remove('epl_matches.db')
        db = MatchDatabase()
        print("✓ Database cleared and reinitialized")

# Load historical data
print("\nLoading historical data into database...")
db.load_historical_data(df)

# Verify
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)

matches = db.get_all_matches()
print(f"\n✓ Total matches in database: {len(matches)}")
print(f"✓ Date range: {matches['date'].min().date()} to {matches['date'].max().date()}")

teams = db.get_all_teams()
print(f"✓ Total teams: {len(teams)}")
print(f"\nTeams in database:")
for i, team in enumerate(sorted(teams), 1):
    print(f"  {i}. {team}")

# Test queries
print("\n" + "=" * 60)
print("TESTING DATABASE QUERIES")
print("=" * 60)

test_team = teams[0]
team_matches = db.get_team_matches(test_team)
print(f"\n✓ {test_team}: {len(team_matches)} matches found")
print(f"  Last 5 matches:")
for _, match in team_matches.head(5).iterrows():
    if match['home_team'] == test_team:
        print(f"    {match['date'].date()}: {match['home_team']} {match['home_goals']}-{match['away_goals']} {match['away_team']} ({match['result']})")
    else:
        print(f"    {match['date'].date()}: {match['home_team']} {match['home_goals']}-{match['away_goals']} {match['away_team']} ({match['result']})")

# Test H2H
if len(teams) > 1:
    team1, team2 = teams[0], teams[1]
    h2h = db.get_h2h_matches(team1, team2)
    print(f"\n✓ H2H between {team1} and {team2}: {len(h2h)} matches found")
    if len(h2h) > 0:
        print(f"  Last 3 H2H:")
        for _, match in h2h.head(3).iterrows():
            print(f"    {match['date'].date()}: {match['home_team']} {match['home_goals']}-{match['away_goals']} {match['away_team']}")

print("\n" + "=" * 60)
print("✓ DATABASE SETUP COMPLETE!")
print("=" * 60)
print("\nYou can now run the Streamlit app!")