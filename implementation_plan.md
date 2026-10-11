# Implementation Plan - Increasing Cricket Prediction Accuracy

This plan outlines the steps to improve the prediction accuracy of the cricket match outcome model. The current baseline accuracy is approximately 51.83%.

## User Review Required

> [!IMPORTANT]
> To achieve significant gains, we need to consolidate team names (e.g., Delhi Daredevils and Delhi Capitals). I will map these to their modern equivalents unless you prefer to keep them separate.

> [!TIP]
> Introducing "Recent Form" (last 5 matches) usually provides a big boost in accuracy for sports predictions.

## Proposed Changes

### 1. Data Cleaning & Consolidation
**Goal**: Ensure the model treats historically renamed teams as the same entity.

- **[MODIFY] [preprocess.py](file:///c:/Users/HP/OneDrive/Desktop/project/Project_Cricket_first/src/preprocess.py)**: 
    - Add a `TEAM_MAPPING` dictionary to unify team names.
    - Apply this mapping during the `load_and_clean` step.

### 2. Advanced Feature Engineering
**Goal**: Capture more nuanced patterns in cricket matches.

- **[MODIFY] [preprocess.py](file:///c:/Users/HP/OneDrive/Desktop/project/Project_Cricket_first/src/preprocess.py)**:
    - **Team Form**: Calculate win rates over the last 5-10 matches for each team.
    - **Venue-Specific Win Rate**: Calculate how often each team wins at the specific match venue.
    - **Toss Advantage**: Add a feature combining toss winner and toss decision relative to the venue's historical "bat-first" vs "bowl-first" success.

### 3. Model Optimization
**Goal**: Use state-of-the-art algorithms and fine-tune their performance.

- **[MODIFY] [train.py](file:///c:/Users/HP/OneDrive/Desktop/project/Project_Cricket_first/src/train.py)**:
    - Integrate **XGBoost** or **LightGBM** into the model suite.
    - Implement **Hyperparameter Tuning** using `RandomizedSearchCV` for the best-performing model.
    - Use **K-Fold Cross-Validation** to ensure the accuracy boost is robust and not just overfitting to the test split.

### 4. Evaluation & Reporting
- **[MODIFY] [evaluate.py](file:///c:/Users/HP/OneDrive/Desktop/project/Project_Cricket_first/src/evaluate.py)**:
    - Update charts to show the improvement over the baseline.
    - Generate a feature importance plot to see which new features helped the most.

## Open Questions

- Should I prioritize **XGBoost** or **LightGBM**? XGBoost is generally more robust for this scale, while LightGBM is faster.
- Do you have any specific "recent form" window in mind (e.g., last 5 matches, last 10?)

## Verification Plan

### Automated Tests
- Run `python src/train.py` to train the new models.
- Run `python src/evaluate.py` to compare metrics against the 51.83% baseline.

### Manual Verification
- Verify the prediction outputs in `app.py` use the newly trained best model.
- Check the `reports/` folder for updated performance charts and feature importance plots.
