import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# ============== Import ===================
crashes_wide = pd.read_csv("./data/processed/crashes_wide.csv")
crashes_wide.set_index("CRASH_RECORD_ID", inplace=True)

# find relevant predictors
importances_df_sum = pd.read_csv("./data/results/rf_importances.csv")
importances_df_sum.set_index("Variable", inplace=True)
importances_df_sum.sort_values("Mean", ascending=False)
estimates_df_sum = pd.read_csv("./data/results/lr_regularized.csv")
estimates_df_sum.set_index("Variable", inplace=True)
estimates_df_sum.sort_values("Mean", ascending=False)

# ===================== compute features =======================

# PRIM_CONTRIBUTORY_CAUSE

causes = ['UNABLE TO DETERMINE/OTHER', 'FOLLOWING TOO CLOSELY',
       'FAILING TO YIELD RIGHT-OF-WAY', 'IMPROPER OVERTAKING/PASSING',
       'IMPROPER LANE USAGE', 'FAILING TO REDUCE SPEED TO AVOID CRASH',
       'IMPROPER TURNING/NO SIGNAL', 'IMPROPER BACKING']

crashes_wide["PRIM_CONTRIBUTORY_CAUSE"] = crashes_wide["PRIM_CONTRIBUTORY_CAUSE"].where(
    crashes_wide["PRIM_CONTRIBUTORY_CAUSE"].isin(causes),
    "UNABLE TO DETERMINE/OTHER"
)
crashes_wide["PRIM_CONTRIBUTORY_CAUSE"].value_counts(dropna=False)

# BEAT_OF_OCCURRENCE
crashes_wide["BEAT_OF_OCCURRENCE"] = crashes_wide["BEAT_OF_OCCURRENCE"].astype("Int64")
crashes_wide["BEAT_OF_OCCURRENCE"].value_counts(dropna=False)

# CRASH_MONTH
crashes_wide["CRASH_MONTH"].value_counts(dropna=False)

crashes_wide["CRASH_SEASON"] = crashes_wide["CRASH_MONTH"].replace({
    1: "WINTER",
    2: "WINTER",
    3: "WINTER",
    4: "SPRING",
    5: "SPRING",
    6: "SPRING",
    7: "SUMMER",
    8: "SUMMER",
    9: "SUMMER",
    10: "FALL",
    11: "FALL",
    12: "FALL",
})

# CRASH_DAY_OF_WEEK
crashes_wide["CRASH_DAY_OF_WEEK"] = crashes_wide["CRASH_DAY_OF_WEEK"].replace({
    1: "SUNDAY",
    2: "MONDAY",
    3: "TUESDAY",
    4: "WEDNESDAY",
    5: "THURSDAY",
    6: "FRIDAY",
    7: "SATURDAY",
})
crashes_wide["CRASH_WEEKDAY"] = np.where(
    crashes_wide["CRASH_DAY_OF_WEEK"].isin(["SUNDAY", "SATURDAY"]),
    "WEEKEND",
    "WEEKDAY"
)
crashes_wide["CRASH_DAY_OF_WEEK"].value_counts(dropna=False)
crashes_wide["CRASH_WEEKDAY"].value_counts(dropna=False)

# CRASH_HOUR
crashes_wide["CRASH_DARK"] = np.where(
    crashes_wide["CRASH_HOUR"].between(6, 20),
    "DAY",
    "NIGHT"
)
crashes_wide["CRASH_HOUR"].value_counts(dropna=False)
crashes_wide["CRASH_DARK"].value_counts(dropna=False)

# VEHICLE_TYPE
crashes_wide["VEHICLE_TYPE_0"].value_counts(dropna=False)
crashes_wide["VEHICLE_TYPE_1"].value_counts(dropna=False)

# MANEUVER

maneuvers = ['STRAIGHT AHEAD', 'SLOW/STOP IN TRAFFIC', 'TURNING LEFT',
       'BACKING', 'TURNING RIGHT', 'CHANGING LANES', 'PASSING/OVERTAKING', 'UNKNOWN/OTHER']

crashes_wide["MANEUVER_0"] = crashes_wide["MANEUVER_0"].where(
    crashes_wide["MANEUVER_0"].isin(maneuvers),
    "UNKNOWN/OTHER"
)
crashes_wide["MANEUVER_1"] = crashes_wide["MANEUVER_1"].where(
    crashes_wide["MANEUVER_1"].isin(maneuvers),
    "UNKNOWN/OTHER"
)
crashes_wide["MANEUVER_0"].value_counts(dropna=False)
crashes_wide["MANEUVER_1"].value_counts(dropna=False)

# SEX
crashes_wide["SEX_0"] = crashes_wide["SEX_0"].replace({
    c: "UNKOWN/OTHER" for c in ["X", "U"]
})
crashes_wide["SEX_1"] = crashes_wide["SEX_1"].replace({
    c: "UNKOWN/OTHER" for c in ["X", "U"]
})
crashes_wide["SEX_0"].fillna("UNKOWN/OTHER", inplace=True)
crashes_wide["SEX_1"].fillna("UNKOWN/OTHER", inplace=True)

crashes_wide["SEX_0"].value_counts(dropna=False)
crashes_wide["SEX_1"].value_counts(dropna=False)
# AGE
crashes_wide["AGE_0"].value_counts(dropna=False)
crashes_wide["AGE_1"].value_counts(dropna=False)

bins = [15, 20, 60, 100]

crashes_wide["AGE_BINNED_0"] = pd.cut(crashes_wide["AGE_0"], bins)
crashes_wide["AGE_BINNED_1"] = pd.cut(crashes_wide["AGE_1"], bins)

crashes_wide["AGE_BINNED_0"] = crashes_wide["AGE_BINNED_0"].astype(str).replace({"nan": "UNKNOWN"})
crashes_wide["AGE_BINNED_1"] = crashes_wide["AGE_BINNED_1"].astype(str).replace({"nan": "UNKNOWN"})

crashes_wide["AGE_BINNED_0"].value_counts(dropna=False)
crashes_wide["AGE_BINNED_1"].value_counts(dropna=False)
# SAFETY_EQUIPMENT
equip = {
    'SAFETY BELT USED': 'SAFETY BELT USED',
    'USAGE UNKNOWN': 'UNKNOWN/OTHER',
    'NONE PRESENT': 'NOT USED/NONE PRESENT',
    'SAFETY BELT NOT USED': 'NOT USED/NONE PRESENT',
    'HELMET NOT USED': 'UNKNOWN/OTHER',
    'HELMET USED': 'UNKNOWN/OTHER',
    'DOT COMPLIANT MOTORCYCLE HELMET': 'UNKNOWN/OTHER',
    'NOT DOT COMPLIANT MOTORCYCLE HELMET': 'UNKNOWN/OTHER',
    'SHOULD/LAP BELT USED IMPROPERLY': 'NOT USED/NONE PRESENT'
}

crashes_wide["SAFETY_EQUIPMENT_0"] = crashes_wide["SAFETY_EQUIPMENT_0"].replace(equip)
crashes_wide["SAFETY_EQUIPMENT_1"] = crashes_wide["SAFETY_EQUIPMENT_1"].replace(equip)

crashes_wide["SAFETY_EQUIPMENT_0"].value_counts(dropna=False)
crashes_wide["SAFETY_EQUIPMENT_1"].value_counts(dropna=False)

# AIRBAG_DEPLOYED_0
airbag = {
    'DID NOT DEPLOY': 'DID NOT DEPLOY',
    'DEPLOYMENT UNKNOWN': 'UNKNOWN/NA',
    'NOT APPLICABLE': 'UNKNOWN/NA',
    'DEPLOYED, FRONT': 'DEPLOYED',
    'DEPLOYED, COMBINATION': 'DEPLOYED',
    'DEPLOYED, SIDE': 'DEPLOYED',
    'DEPLOYED OTHER (KNEE, AIR, BELT, ETC.)': 'DEPLOYED'
}

crashes_wide["AIRBAG_DEPLOYED_0"] = crashes_wide["AIRBAG_DEPLOYED_0"].replace(airbag)
crashes_wide["AIRBAG_DEPLOYED_1"] = crashes_wide["AIRBAG_DEPLOYED_1"].replace(airbag)

crashes_wide["AIRBAG_DEPLOYED_0"].value_counts(dropna=False)
crashes_wide["AIRBAG_DEPLOYED_1"].value_counts(dropna=False)

# DRIVER_ACTION

actions = ['NONE', 'FAILED TO YIELD', 'UNKNOWN/OTHER',
       'FOLLOWED TOO CLOSELY', 'IMPROPER BACKING', 'IMPROPER LANE CHANGE',
       'IMPROPER TURN', 'IMPROPER PASSING', 'TOO FAST FOR CONDITIONS',
       'DISREGARDED CONTROL DEVICES']

crashes_wide["DRIVER_ACTION_0"] = crashes_wide["DRIVER_ACTION_0"].where(
    crashes_wide["DRIVER_ACTION_0"].isin(actions),
    "UNKNOWN/OTHER"
)
crashes_wide["DRIVER_ACTION_1"] = crashes_wide["DRIVER_ACTION_1"].where(
    crashes_wide["DRIVER_ACTION_1"].isin(actions),
    "UNKNOWN/OTHER"
)
crashes_wide["DRIVER_ACTION_0"].value_counts(dropna=False)
crashes_wide["DRIVER_ACTION_1"].value_counts(dropna=False)

# DRIVER_ACTION

test = {
    "TEST NOT OFFERED": "TEST NOT OFFERED",
    "TEST REFUSED": "TEST OFFERED",
    "TEST PERFORMED, RESULTS UNKNOWN": "TEST OFFERED",
    "TEST TAKEN": "TEST OFFERED"
}

crashes_wide["BAC_RESULT_0"] = crashes_wide["BAC_RESULT_0"].replace(test)
crashes_wide["BAC_RESULT_1"] = crashes_wide["BAC_RESULT_1"].replace(test)

crashes_wide["BAC_RESULT_0"].value_counts(dropna=False)
crashes_wide["BAC_RESULT_1"].value_counts(dropna=False)


# ============ list of features to consider ================
features = [
    'PRIM_CONTRIBUTORY_CAUSE',
    'BEAT_OF_OCCURRENCE',
    'CRASH_MONTH', 'CRASH_SEASON',
    'CRASH_DAY_OF_WEEK', 'CRASH_WEEKDAY',
    'CRASH_HOUR', 'CRASH_DARK',
    'VEHICLE_TYPE_0', 'VEHICLE_TYPE_1',
    'MANEUVER_0', 'MANEUVER_1',
    "OCCUPANT_CNT_0", "OCCUPANT_CNT_1",
    'SEX_0', 'SEX_1',
    'AGE_0', 'AGE_1', 'AGE_BINNED_0', 'AGE_BINNED_1',
    'SAFETY_EQUIPMENT_0', 'SAFETY_EQUIPMENT_1',
    'AIRBAG_DEPLOYED_0', 'AIRBAG_DEPLOYED_1',
    'DRIVER_ACTION_0', 'DRIVER_ACTION_1',
    'BAC_RESULT_0', 'BAC_RESULT_1'
]
# ============ compute responses ================
crashes_wide["INJURIES_0_ANY"] = crashes_wide["INJURY_CLASSIFICATION_0"].ne("NO INDICATION OF INJURY")
crashes_wide["INJURIES_0_SEVERE"] = crashes_wide["INJURY_CLASSIFICATION_0"].isin(
    ["FATAL", "INCAPACITATING INJURY"]
)

responses = ["INJURIES_0_ANY", "INJURIES_0_SEVERE"]




crashes_final = crashes_wide[responses + features]
# crashes_final = crashes_wide[responses + features].dropna(axis=0)

crashes_final.to_csv("./data/processed/crashes_final.csv")