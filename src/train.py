import os
import sys
import warnings
import joblib
from sklearn.model_selection   import train_test_split
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics           import accuracy_score
from sklearn.pipeline          import Pipeline
from sklearn.preprocessing     import StandardScaler

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*converge.*")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import (
    load_and_clean, compute_team_stats, save_team_stats,
    add_features, encode, save_encoders,
    get_features_and_target, save_processed_csv, MODELS_DIR
)

MODELS = {
    "Logistic Regression" : Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(max_iter=3000, random_state=42)),
    ]),
    "Random Forest"       : RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
    ),
    "Gradient Boosting"   : GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.85,
        random_state=42,
    ),
}


def train():
    df_clean    = load_and_clean()
    stats       = compute_team_stats(df_clean)
    save_team_stats()
    df_featured = add_features(df_clean, stats)
    save_processed_csv(df_featured)
    df_encoded  = encode(df_featured, fit=True)
    save_encoders()

    X, y = get_features_and_target(df_encoded)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n[INFO] Train rows : {len(X_train)}")
    print(f"[INFO] Test  rows : {len(X_test)}")

    results        = {}
    trained_models = {}

    print("\n-- Training all 3 models --")
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)

        results[name]        = acc
        trained_models[name] = model

        bar = "#" * int(acc * 30)
        print(f"  {name:<25}  {acc*100:5.1f}%  {bar}")

    best_name  = max(results, key=results.get)
    best_model = trained_models[best_name]

    print(f"\n[BEST] '{best_name}'  with  {results[best_name]*100:.1f}% accuracy")

    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"[INFO] Best model saved  ->  {model_path}")

    all_path = os.path.join(MODELS_DIR, "all_models.pkl")
    joblib.dump(trained_models, all_path)
    print(f"[INFO] All models saved  ->  {all_path}")

    test_path = os.path.join(MODELS_DIR, "test_data.pkl")
    joblib.dump({"X_test": X_test, "y_test": y_test}, test_path)
    print(f"[INFO] Test data saved   ->  {test_path}")

    return trained_models, X_test, y_test


if __name__ == "__main__":
    train()
    print("\n[OK] train.py")