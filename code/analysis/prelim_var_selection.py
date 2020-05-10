import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.impute import KNNImputer

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# import
crashes_wide = pd.read_csv("./data/processed/crashes_wide.csv")
crashes_wide.set_index("CRASH_RECORD_ID", inplace=True)

# prepare
features = ['POSTED_SPEED_LIMIT', 'TRAFFIC_CONTROL_DEVICE',
       'DEVICE_CONDITION', 'WEATHER_CONDITION', 'LIGHTING_CONDITION',
       'FIRST_CRASH_TYPE', 'TRAFFICWAY_TYPE', 'ALIGNMENT',
       'ROADWAY_SURFACE_COND', 'ROAD_DEFECT',
       'PRIM_CONTRIBUTORY_CAUSE', 'SEC_CONTRIBUTORY_CAUSE',
       'STREET_DIRECTION', 'BEAT_OF_OCCURRENCE', 'CRASH_HOUR',
       'CRASH_DAY_OF_WEEK', 'CRASH_MONTH',
       'VEHICLE_DEFECT_0', 'VEHICLE_DEFECT_1', 'VEHICLE_TYPE_0',
       'VEHICLE_TYPE_1', 'TRAVEL_DIRECTION_0', 'TRAVEL_DIRECTION_1',
       'MANEUVER_0', 'MANEUVER_1', 'OCCUPANT_CNT_0', 'OCCUPANT_CNT_1',
       'SEX_0', 'SEX_1', 'AGE_0', 'AGE_1',
       'SAFETY_EQUIPMENT_0', 'SAFETY_EQUIPMENT_1',
            #'AIRBAG_DEPLOYED_0', 'AIRBAG_DEPLOYED_1',
            'EJECTION_0', 'EJECTION_1',
       'DRIVER_ACTION_0', 'DRIVER_ACTION_1', 'DRIVER_VISION_0',
       'DRIVER_VISION_1', 'BAC_RESULT_0', 'BAC_RESULT_1']

design_matrix = pd.DataFrame(index=crashes_wide.index)

for col in features:
    x = crashes_wide[col]
    if np.issubdtype(x, np.number):
        design_matrix[col] = x
    else:
        X = pd.get_dummies(x)
        X.loc[x.isna()] = np.nan
        for c in X.columns:
            design_matrix[col + "[" + c + "]"] = X[c]

crashes_wide.isna().sum()
design_matrix.isna().sum()
X = design_matrix.to_numpy()

# warning: takes a few minutes
imputer = KNNImputer(n_neighbors=10, weights="uniform")
X = imputer.fit_transform(X)

crashes_wide["INJURIES_ALL"] = crashes_wide["INJURY_CLASSIFICATION_0"].ne("NO INDICATION OF INJURY")
crashes_wide["INJURIES_SEVERE"] = crashes_wide["INJURY_CLASSIFICATION_0"].isin(
    ["FATAL", "INCAPACITATING INJURY", "NONINCAPACITATING INJURY"]
)

responses = ["INJURIES_ALL", "INJURIES_SEVERE"]

importances = dict()

for response in responses:
    y = crashes_wide[response].to_numpy()
    labeler = LabelBinarizer(0, 1)
    y = labeler.fit_transform(y)

    # model
    rf = RandomForestClassifier(class_weight="balanced")
    rf.fit(X, y.ravel())

    cols = design_matrix.columns
    cols_parsed = []
    for col in cols:
        sq = col.find("[")
        if sq > 0:
            var = col[:sq]
            val = col[(sq + 1):-1]
        else:
            var = col
            val = ""
        cols_parsed.append((var, val))

    var_importance = pd.DataFrame({
        "Variable": pd.MultiIndex.from_tuples(cols_parsed).get_level_values(0),
        "Level": pd.MultiIndex.from_tuples(cols_parsed).get_level_values(1),
        "Importance": rf.feature_importances_
    })
    var_importance_sum = var_importance.groupby("Variable").agg("sum")
    importances[response] = var_importance_sum

importances_df = pd.DataFrame(index=importances[responses[0]].index)
for response, imp in importances.items():
    importances_df[response] = imp

# Sum 0/1

vars = importances_df.index.values
vars_parsed = []
for var in vars:
    if var.endswith("0") or var.endswith("1"):
        val = var[-1]
        var = var[:-2]
    else:
        val = ""
    vars_parsed.append((var, val))

importances_df.index = pd.MultiIndex.from_tuples(vars_parsed)
importances_df.reset_index(inplace=True)
importances_df_sum = importances_df.groupby("level_0").agg("sum")

importances_df_sum["Mean"] = importances_df_sum.mean(1)

importances_df_sum.sort_values("Mean", ascending=False)
importances_df_sum.index.name = "Variable"

importances_df_sum.to_csv("./data/results/rf_importances.csv")



# Group Lasso Logistic Regression

features_wide = design_matrix.columns

groups = ["sdfs"]
groups_ids = []

n = 0
for col in features_wide:
    if col.find("[") > 0:
        groups.append(col[:col.find("[")])
    else:
        groups.append(col)
    if groups[-1][:-2] != groups[-2][:-2]:
        n += 1
    groups_ids.append(n)
groups = groups[1:]

scaler = StandardScaler()
Xstd = scaler.fit_transform(X=X)

Cs = [0.001]

estimates = dict()

for response in responses:
    y = crashes_wide[response].to_numpy()
    labeler = LabelBinarizer(0, 1)
    y = labeler.fit_transform(y)

    # model
    lr = LogisticRegressionCV(
        Cs=Cs,
        cv=5,
        penalty="elasticnet",
        l1_ratios=[0.95]*len(Cs),
        scoring="roc_auc",
        class_weight="balanced",
        solver="saga",
        n_jobs=-1
    )
    lr.fit(Xstd, y.ravel())



    cols = design_matrix.columns
    cols_parsed = []
    for col in cols:
        sq = col.find("[")
        if sq > 0:
            var = col[:sq]
            val = col[(sq + 1):-1]
        else:
            var = col
            val = ""
        cols_parsed.append((var, val))

    coefs = pd.DataFrame({
        "Variable": pd.MultiIndex.from_tuples(cols_parsed).get_level_values(0),
        "Level": pd.MultiIndex.from_tuples(cols_parsed).get_level_values(1),
        "Estimate": lr.coef_.ravel(),
        "Included": np.abs(lr.coef_).ravel() > 0
    })
    estimates_mean = coefs.groupby("Variable").agg({
        "Estimate": lambda x: np.sqrt(np.sum(x**2) / len(x)),
        "Included": np.mean
    })
    estimates[response] = estimates_mean


estimates_df = pd.DataFrame(index=estimates[responses[0]].index)
for response, imp in estimates.items():
    estimates_df[response] = imp["Estimate"]

# Sum 0/1

vars = estimates_df.index.values
vars_parsed = []
for var in vars:
    if var.endswith("0") or var.endswith("1"):
        val = var[-1]
        var = var[:-2]
    else:
        val = ""
    vars_parsed.append((var, val))

estimates_df.index = pd.MultiIndex.from_tuples(vars_parsed)
estimates_df.reset_index(inplace=True)
estimates_df_sum = estimates_df.groupby("level_0").agg(lambda x: np.sqrt(np.sum(x**2) / len(x)))

estimates_df_sum["Mean"] = estimates_df_sum.mean(1)

estimates_df_sum.sort_values("Mean", ascending=False)
estimates_df_sum.index.name = "Variable"
estimates_df_sum.to_csv("./data/results/lr_regularized.csv")