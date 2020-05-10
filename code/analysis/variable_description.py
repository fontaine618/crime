import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.gam.api import BSplines, GLMGam
import patsy
import itertools
import sys
sys.path.append("/home/simon/Documents/sithon")
from sithon.model_output import coef_table_long, coef_table_wide

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width', 1000)

# ============== Import ===================
crashes = pd.read_csv("./data/processed/crashes_final.csv")
crashes.set_index("CRASH_RECORD_ID", inplace=True)

vars = [
    'INJURIES_0_ANY', 'INJURIES_0_SEVERE', 'PRIM_CONTRIBUTORY_CAUSE',
    'BEAT_OF_OCCURRENCE', 'CRASH_MONTH', 'CRASH_SEASON', 'CRASH_DAY_OF_WEEK',
    'CRASH_WEEKDAY', 'CRASH_HOUR', 'CRASH_DARK', 'VEHICLE_TYPE_0',
    'VEHICLE_TYPE_1', 'MANEUVER_0', 'MANEUVER_1', 'SEX_0', 'SEX_1', 'AGE_0',
    'AGE_1', 'AGE_BINNED_0', 'AGE_BINNED_1', 'SAFETY_EQUIPMENT_0',
    'SAFETY_EQUIPMENT_1', 'AIRBAG_DEPLOYED_0', 'AIRBAG_DEPLOYED_1',
    'DRIVER_ACTION_0', 'DRIVER_ACTION_1', 'BAC_RESULT_0', 'BAC_RESULT_1'
]


# ============== CRASH ===================

vars_crash = [
    'PRIM_CONTRIBUTORY_CAUSE',
    'CRASH_MONTH', 'CRASH_HOUR', 'CRASH_DAY_OF_WEEK',
    'CRASH_SEASON', 'CRASH_WEEKDAY', 'CRASH_DARK',
]

vars_crash_calc = [False] * 4 + [True] * 3

vars_crash_values = [
    "\\newline ".join([
        "{} ({})".format(value, count)
        for value, count
        in crashes[var].value_counts().iteritems()])
    for var in vars_crash
]

vars_crash_values[1] = "1-12"
vars_crash_values[2] = "0-23"
vars_crash_values[3] = "SUNDAY-SATURDAY"

vars_crash_comments = [
    "NA and low frequency levels assigned to UNABLE TO DETERMINE/OTHER.",
    "",
    "",
    "",
    "",
    "WEEKEND defined as SATURDAY and SUNDAY",
    "DAY defined between 6AM and 8PM"
]

tab_crash = pd.DataFrame({
    "Unit": "Crash",
    "Description": "",
    "Calculated": vars_crash_calc,
    "Values": vars_crash_values,
    "Comments": vars_crash_comments
}, index=vars_crash)

tab_crash.to_csv("./data/results/vars_crash.csv")

# ============== VEHICLE ===================

vars_vehicle = [
    'VEHICLE_TYPE_0', 'VEHICLE_TYPE_1', 'MANEUVER_0', 'MANEUVER_1'
]

vars_vehicle_calc = [False] * 4

vars_vehicle_values = [
    "\\newline ".join([
        "{} ({})".format(value, count)
        for value, count
        in crashes[var].value_counts().iteritems()])
    for var in vars_vehicle
]

vars_vehicle_comments = [
    "",
    "",
    "NA and low frequency levels assigned to UNKNOWN/OTHER",
    "NA and low frequency levels assigned to UNKNOWN/OTHER",
]

tab_vehicle = pd.DataFrame({
    "Unit": "Vehicle",
    "Description": "",
    "Calculated": vars_vehicle_calc,
    "Values": vars_vehicle_values,
    "Comments": vars_vehicle_comments
}, index=vars_vehicle)

tab_vehicle.to_csv("./data/results/vars_vehicle.csv")

# ============== DRIVERS ===================

vars_drivers = [
    'SEX_0', 'SEX_1', 'SAFETY_EQUIPMENT_0',
    'SAFETY_EQUIPMENT_1', 'AIRBAG_DEPLOYED_0', 'AIRBAG_DEPLOYED_1',
    'DRIVER_ACTION_0', 'DRIVER_ACTION_1', 'BAC_RESULT_0', 'BAC_RESULT_1',
    'AGE_BINNED_0', 'AGE_BINNED_1',
    'INJURIES_0_ANY', 'INJURIES_0_SEVERE',
]

vars_drivers_calc = [False] * 10 + [True] * 4

vars_drivers_values = [
    "\\newline ".join([
        "{} ({})".format(value, count)
        for value, count
        in crashes[var].value_counts().iteritems()])
    for var in vars_drivers
]

vars_drivers_comments = [
    "NA and U assigned to UNKNOWN/OTHER",
    "NA and U assigned to UNKNOWN/OTHER",
    "NA and low frequency levels assigned to UNKNOWN/OTHER",
    "NA and low frequency levels assigned to UNKNOWN/OTHER",
    "NA and low frequency levels assigned to UNKNOWN/NA",
    "NA and low frequency levels assigned to UNKNOWN/NA",
    "NA and low frequency levels assigned to UNKNOWN/OTHER",
    "NA and low frequency levels assigned to UNKNOWN/OTHER",
    "TEST REFUSED, TEST PERFORMED, RESULTS UNKNOWN and TEST TAKEN assigned to TEST OFFERED",
    "TEST REFUSED, TEST PERFORMED, RESULTS UNKNOWN and TEST TAKEN assigned to TEST OFFERED",
    "Computed from AGE",
    "Computed from AGE",
    "True if not NO INDICATION OF INJURY",
    "True if FATAL or INCAPACITATING INJURY",
]

tab_drivers = pd.DataFrame({
    "Unit": "Drivers",
    "Description": "",
    "Calculated": vars_drivers_calc,
    "Values": vars_drivers_values,
    "Comments": vars_drivers_comments
}, index=vars_drivers)

tab_drivers.to_csv("./data/results/vars_drivers.csv")



print(crashes.columns.values)


# ============= OCCUPANTS =================

occupant_type = pd.DataFrame({
    "OCCUPANT_CNT": pd.concat([
        crashes["OCCUPANT_CNT_0"],
        crashes["OCCUPANT_CNT_1"]
    ]).astype(int),
    "VEHICLE_TYPE": pd.concat([
        crashes["VEHICLE_TYPE_0"],
        crashes["VEHICLE_TYPE_1"]
    ])
})

ctab = pd.crosstab(
    index=occupant_type["OCCUPANT_CNT"],
    columns=occupant_type["VEHICLE_TYPE"]
)

ctab.drop(index=[0], inplace=True)

ctab.loc["TOTAL"] = ctab.sum()

ftab = 100 * ctab / ctab.loc["TOTAL"]

ftab = ftab.applymap("{:.2f}".format)

occupant_type.groupby("VEHICLE_TYPE").agg("mean")