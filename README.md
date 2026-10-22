# Decision Trees with scikit-learn

Two Jupyter notebooks that walk through decision trees in scikit-learn: one for **classification** (Titanic survival) and one for **regression** (diabetes progression). The classifier notebook also covers pre-pruning and post-pruning to control overfitting.

## Contents

| Notebook | Task | Dataset | Target |
|---|---|---|---|
| `decision_tree_classifier.ipynb` | Classification | Titanic (`seaborn.load_dataset("titanic")`) | `survived` (0 = Died, 1 = Survived) |
| `decision_tree_regressor.ipynb` | Regression | Diabetes (`sklearn.datasets.load_diabetes`) | `target` (disease progression measure) |

## 1. Decision Tree Classifier

**Features used:** `pclass`, `sex`, `fare`, `embarked`, `age`

**Workflow**

1. **Load data**: Titanic dataset from seaborn; inspect missing values.
2. **Handle missing values** with `SimpleImputer`:
   - `age`: median
   - `embarked`: most frequent value
3. **Encode categoricals** (`sex`, `embarked`) with `LabelEncoder`.
4. **Split**: 70% train / 30% test (`random_state=42`).
5. **Baseline model**: unpruned `DecisionTreeClassifier`, evaluated with accuracy and visualized with `plot_tree` (top 2 levels).
6. **Pre-pruning** (limit growth while training):
   - Sweep `max_depth` from 2 to 10 and compare test accuracy (tree plotted at depth 4).
   - Sweep `min_samples_split` from 5 to 30 with `max_depth=4` (tree plotted at 10).
7. **Post-pruning** (grow fully, then trim):
   - Compute the cost-complexity pruning path (`cost_complexity_pruning_path`) to get candidate `ccp_alpha` values.
   - Train one tree per alpha, pick the alpha with the best test accuracy.
   - Fit a final model using `ccp_alpha=best_alpha` and `max_depth=4`, then plot it and report its accuracy.

**Key concepts:** overfitting in unpruned trees, pre-pruning vs. post-pruning, cost-complexity pruning, tree visualization.

## 2. Decision Tree Regressor

**Features:** all 10 numeric features of the diabetes dataset (age, sex, bmi, bp, s1–s6).

**Workflow**

1. Load the dataset as a DataFrame and separate `target`.
2. Split 70% train / 30% test (`random_state=42`).
3. Train `DecisionTreeRegressor(max_depth=7, min_samples_leaf=20)`; both settings limit tree complexity.
4. Evaluate on train and test sets using **MSE** and **R²** to check for overfitting (a large train/test gap signals overfitting).
5. Visualize the tree with `plot_tree`.

## Requirements

- Python 3.8+
- `scikit-learn`
- `pandas`
- `matplotlib`
- `seaborn`
- `jupyter` (or JupyterLab / VS Code with the Jupyter extension)

Install everything with:

```bash
pip install scikit-learn pandas matplotlib seaborn jupyter
```

## Running the notebooks

```bash
jupyter notebook
```

Open either notebook and run the cells top to bottom. The Titanic dataset is downloaded by seaborn on first use, so an internet connection is needed for the classifier notebook; the diabetes dataset ships with scikit-learn.

## Notes and possible improvements

- Model selection in the classifier notebook (best depth, split size, and alpha) is done against the **test set**. For a more reliable estimate, use a separate validation set or cross-validation (e.g. `GridSearchCV`) and keep the test set for the final evaluation only.
- `LabelEncoder` is applied to input features; `OneHotEncoder` or `OrdinalEncoder` is generally preferred for features, since label encoding implies an order. Trees tolerate this better than linear models do.
- The imputers and encoders are fit on the full dataset before the train/test split, which can leak information. Wrapping them in a `Pipeline` and fitting on the training data only avoids this.
- Some models in the classifier notebook have no `random_state`, so results may vary slightly between runs.
- Try tuning the regressor (`max_depth`, `min_samples_leaf`, `ccp_alpha`) and compare against a baseline such as linear regression or a random forest.

## License

Add your preferred license here (e.g. MIT).
