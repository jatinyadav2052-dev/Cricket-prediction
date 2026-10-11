import os
import sys
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "matches.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

CAT_COLS = ["team1", "team2", "toss_winner", "toss_decision", "venue"]

FEATURE_COLS = [
    "team1", "team2", "toss_winner", "toss_decision", "venue",
    "team1_win_rate",
    "team2_win_rate",
    "win_rate_diff",
    "h2h_win_rate",
    "home_adv",
]

TARGET_COL = "winner"

encoders = {}

team_stats = {}

HOME_VENUES = {
    "Chennai Super Kings"        : "MA Chidambaram Stadium, Chepauk",
    "Deccan Chargers"            : "Rajiv Gandhi International Stadium, Uppal",
    "Delhi Capitals"             : "Arun Jaitley Stadium",
    "Delhi Daredevils"           : "Feroz Shah Kotla",
    "Gujarat Lions"              : "Saurashtra Cricket Association Stadium",
    "Gujarat Titans"             : "Narendra Modi Stadium, Ahmedabad",
    "Kings XI Punjab"            : "Punjab Cricket Association Stadium, Mohali",
    "Kochi Tuskers Kerala"       : "Nehru Stadium",
    "Kolkata Knight Riders"      : "Eden Gardens",
    "Lucknow Super Giants"       : "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
    "Mumbai Indians"             : "Wankhede Stadium",
    "Pune Warriors"              : "Subrata Roy Sahara Stadium",
    "Punjab Kings"               : "Punjab Cricket Association IS Bindra Stadium, Mohali",
    "Rajasthan Royals"           : "Sawai Mansingh Stadium",
    "Rising Pune Supergiant"     : "Maharashtra Cricket Association Stadium",
    "Rising Pune Supergiants"    : "Maharashtra Cricket Association Stadium",
    "Royal Challengers Bangalore": "M Chinnaswamy Stadium",
    "Royal Challengers Bengaluru": "M Chinnaswamy Stadium, Bengaluru",
    "Sunrisers Hyderabad"        : "Rajiv Gandhi International Stadium, Uppal",
}


def load_and_clean(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df)} rows from {path}")

    df = df[["team1", "team2", "toss_winner", "toss_decision", "venue", TARGET_COL]].copy()

    before = len(df)
    df.dropna(inplace=True)
    print(f"[INFO] Dropped {before - len(df)} null rows  ->  {len(df)} rows remain")

    return df


def compute_team_stats(df):
    global team_stats

    all_teams = set(df["team1"].tolist() + df["team2"].tolist())
    match_count = {t: 0 for t in all_teams}
    win_count   = {t: 0 for t in all_teams}

    for _, row in df.iterrows():
        t1, t2, winner = row["team1"], row["team2"], row["winner"]
        match_count[t1] = match_count.get(t1, 0) + 1
        match_count[t2] = match_count.get(t2, 0) + 1
        if winner in match_count:
            win_count[winner] = win_count.get(winner, 0) + 1

    win_rates = {
        t: win_count[t] / match_count[t] if match_count[t] > 0 else 0.5
        for t in all_teams
    }

    h2h_wins  = {}
    h2h_total = {}

    for _, row in df.iterrows():
        t1, t2, winner = row["team1"], row["team2"], row["winner"]
        for pair in [(t1, t2), (t2, t1)]:
            h2h_total[pair] = h2h_total.get(pair, 0) + 1

        if winner == t1:
            h2h_wins[(t1, t2)] = h2h_wins.get((t1, t2), 0) + 1
        elif winner == t2:
            h2h_wins[(t2, t1)] = h2h_wins.get((t2, t1), 0) + 1

    team_stats = {
        "win_rates" : win_rates,
        "h2h_wins"  : h2h_wins,
        "h2h_total" : h2h_total,
    }

    print(f"[INFO] Win rates computed for {len(win_rates)} teams")
    print(f"[INFO] H2H records computed for {len(h2h_total)//2} unique matchups")
    return team_stats


def add_features(df, stats=None):
    if stats is None:
        stats = team_stats

    win_rates = stats["win_rates"]
    h2h_wins  = stats["h2h_wins"]
    h2h_total = stats["h2h_total"]

    df = df.copy()

    df["team1_win_rate"] = df["team1"].map(lambda t: win_rates.get(t, 0.5))
    df["team2_win_rate"] = df["team2"].map(lambda t: win_rates.get(t, 0.5))

    df["win_rate_diff"] = df["team1_win_rate"] - df["team2_win_rate"]

    def h2h_rate(row):
        t1, t2 = row["team1"], row["team2"]
        total = h2h_total.get((t1, t2), 0)
        if total == 0:
            return 0.5
        wins = h2h_wins.get((t1, t2), 0)
        return wins / total

    df["h2h_win_rate"] = df.apply(h2h_rate, axis=1)

    df["home_adv"] = df.apply(
        lambda r: 1 if HOME_VENUES.get(r["team1"]) == r["venue"] else 0,
        axis=1
    )

    print(f"[INFO] team1_win_rate range  : [{df['team1_win_rate'].min():.3f}, {df['team1_win_rate'].max():.3f}]")
    print(f"[INFO] win_rate_diff range   : [{df['win_rate_diff'].min():.3f}, {df['win_rate_diff'].max():.3f}]")
    print(f"[INFO] h2h_win_rate range    : [{df['h2h_win_rate'].min():.3f}, {df['h2h_win_rate'].max():.3f}]")
    print(f"[INFO] home_adv (team1 home) : {df['home_adv'].sum()} / {len(df)} rows")

    return df


def save_team_stats():
    path = os.path.join(MODELS_DIR, "team_stats.pkl")
    joblib.dump(team_stats, path)
    print(f"[INFO] Team stats saved  ->  {path}")


def load_team_stats():
    global team_stats
    path = os.path.join(MODELS_DIR, "team_stats.pkl")
    team_stats = joblib.load(path)
    print(f"[INFO] Team stats loaded from {path}")
    return team_stats


def encode(df, fit=True):
    df_enc = df.copy()

    for col in CAT_COLS + [TARGET_COL]:
        if fit:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df_enc[col] = df_enc[col].astype(str).map(
                lambda val, le=le: (
                    le.transform([val])[0]
                    if val in le.classes_
                    else -1
                )
            )
    return df_enc


def get_features_and_target(df_encoded):
    X = df_encoded[FEATURE_COLS]
    y = df_encoded[TARGET_COL]
    return X, y


def save_encoders():
    path = os.path.join(MODELS_DIR, "label_encoders.pkl")
    joblib.dump(encoders, path)
    print(f"[INFO] Encoders saved  ->  {path}")


def load_encoders():
    global encoders
    path = os.path.join(MODELS_DIR, "label_encoders.pkl")
    encoders = joblib.load(path)
    print(f"[INFO] Encoders loaded from {path}")


def save_processed_csv(df_clean):
    out_path = os.path.join(BASE_DIR, "data", "processed.csv")
    df_clean.to_csv(out_path, index=False)
    print(f"[INFO] Processed CSV saved  ->  {out_path}")


if __name__ == "__main__":
    df_clean    = load_and_clean()
    stats       = compute_team_stats(df_clean)
    df_featured = add_features(df_clean, stats)
    save_processed_csv(df_featured)
    save_team_stats()

    df_encoded = encode(df_featured, fit=True)
    save_encoders()

    X, y = get_features_and_target(df_encoded)

    print("\n-- First 5 encoded rows --")
    print(df_encoded[FEATURE_COLS].head().to_string())
    print(f"\nX shape : {X.shape}   (rows, features)")
    print(f"y unique: {y.nunique()} different winners")
    print("\n[OK] preprocess.py")