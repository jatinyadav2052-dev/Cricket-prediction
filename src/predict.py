import os
import sys
import joblib
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import (
    load_encoders, load_team_stats,
    MODELS_DIR, FEATURE_COLS, CAT_COLS, HOME_VENUES
)

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")


def _encode_input(input_df):
    from preprocess import encoders
    enc = input_df.copy()
    for col in CAT_COLS:
        le  = encoders[col]
        val = str(enc.at[0, col])
        enc[col] = int(le.transform([val])[0]) if val in le.classes_ else -1
    return enc


def _build_input_row(team1, team2, toss_winner, toss_decision, venue):
    from preprocess import team_stats

    win_rates = team_stats.get("win_rates", {})
    h2h_wins  = team_stats.get("h2h_wins",  {})
    h2h_total = team_stats.get("h2h_total", {})

    t1_wr = win_rates.get(team1, 0.5)
    t2_wr = win_rates.get(team2, 0.5)

    total = h2h_total.get((team1, team2), 0)
    wins  = h2h_wins.get((team1, team2), 0)
    h2h_wr = wins / total if total > 0 else 0.5

    home_adv = 1 if HOME_VENUES.get(team1) == venue else 0

    return pd.DataFrame([{
        "team1"          : team1,
        "team2"          : team2,
        "toss_winner"    : toss_winner,
        "toss_decision"  : toss_decision,
        "venue"          : venue,
        "team1_win_rate" : t1_wr,
        "team2_win_rate" : t2_wr,
        "win_rate_diff"  : t1_wr - t2_wr,
        "h2h_win_rate"   : h2h_wr,
        "home_adv"       : home_adv,
    }])


def load_model():
    model = joblib.load(MODEL_PATH)
    load_encoders()
    load_team_stats()
    return model


def predict_winner(team1, team2, toss_winner, toss_decision, venue):
    model = load_model()
    from preprocess import encoders

    input_df     = _build_input_row(team1, team2, toss_winner, toss_decision, venue)
    input_enc    = _encode_input(input_df)
    pred_encoded = model.predict(input_enc[FEATURE_COLS])
    winner       = encoders["winner"].inverse_transform(pred_encoded)[0]
    return winner


def predict_with_proba(team1, team2, toss_winner, toss_decision, venue):
    model = load_model()
    from preprocess import encoders

    input_df  = _build_input_row(team1, team2, toss_winner, toss_decision, venue)
    input_enc = _encode_input(input_df)

    proba   = model.predict_proba(input_enc[FEATURE_COLS])[0]
    classes = encoders["winner"].inverse_transform(model.classes_)

    proba_dict = dict(sorted(
        zip(classes, proba * 100),
        key=lambda x: x[1], reverse=True
    ))
    winner = max(proba_dict, key=proba_dict.get)
    return winner, proba_dict


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  IPL MATCH WINNER PREDICTOR")
    print("="*55)

    team1         = "Mumbai Indians"
    team2         = "Chennai Super Kings"
    toss_winner   = "Mumbai Indians"
    toss_decision = "bat"
    venue         = "Wankhede Stadium"

    print(f"\n  Team 1        : {team1}")
    print(f"  Team 2        : {team2}")
    print(f"  Toss winner   : {toss_winner}")
    print(f"  Toss decision : {toss_decision}")
    print(f"  Venue         : {venue}")

    try:
        winner, proba_dict = predict_with_proba(
            team1, team2, toss_winner, toss_decision, venue
        )
        print(f"\n  [WIN] Predicted Winner  ->  {winner}\n")
        print("  Top 5 win probabilities:")
        for i, (team, prob) in enumerate(list(proba_dict.items())[:5], 1):
            bar = "#" * int(prob / 2)
            print(f"  {i}. {team:<32} {prob:5.1f}%  {bar}")

    except Exception as e:
        print(f"\n  [ERROR] {e}")
        print("  -> Run train.py first to generate the model files.")

    print("\n[OK] predict.py")