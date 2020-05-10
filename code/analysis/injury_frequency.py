import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# import
crashes_wide = pd.read_csv("./data/processed/crashes_wide.csv")
crashes_wide.set_index("CRASH_RECORD_ID", inplace=True)

# get only car type and injury report

car_injury = crashes_wide[
    ["VEHICLE_TYPE_0", "VEHICLE_TYPE_1", "INJURY_CLASSIFICATION_0", "INJURY_CLASSIFICATION_1"]
]

car_injury_long = pd.DataFrame({
    "VEHICLE_TYPE": np.concatenate(
        [car_injury["VEHICLE_TYPE_0"].values, car_injury["VEHICLE_TYPE_1"].values]
    ),
    "VEHICLE_TYPE_OTHER": np.concatenate(
        [car_injury["VEHICLE_TYPE_1"].values, car_injury["VEHICLE_TYPE_0"].values]
    ),
    "INJURY_CLASSIFICATION": np.concatenate(
        [car_injury["INJURY_CLASSIFICATION_0"].values, car_injury["INJURY_CLASSIFICATION_1"].values]
    ),
})

ctab = pd.crosstab(
    index=car_injury_long["INJURY_CLASSIFICATION"],
    columns=[
        car_injury_long["VEHICLE_TYPE"],
        car_injury_long["VEHICLE_TYPE_OTHER"],
    ]
)
ctab.loc["TOTAL"] = ctab.sum()

ctab.loc["TOTAL INJURY"] = ctab.loc[[
    "FATAL",
    "INCAPACITATING INJURY",
    "NONINCAPACITATING INJURY",
    "REPORTED, NOT EVIDENT"
]].sum()

ctab.loc["SEVERE INJURY"] = ctab.loc[[
    "FATAL",
    "INCAPACITATING INJURY"
]].sum()

ftab = 100 * ctab / ctab.loc["TOTAL"]

tab = pd.concat([ctab, ftab], 1)
tab = tab.iloc[:, [0, 4, 1, 5, 2, 6, 3, 7]]
tab = tab.iloc[[0, 1, 7, 3, 4, 6, 2, 5]]

tab.columns = pd.MultiIndex.from_product([
    ["PASSENGER", "SUV/PICKUP"],
    ["PASSENGER", "SUV/PICKUP"],
    ["Count", "% Column"],
], names=["VEHICULE_TYPE", "VEHICULE_TYPE_OTHER", ""])

tab.iloc[:, [1, 3, 5, 7]] = tab.iloc[:, [1, 3, 5, 7]].applymap("{:.2f}".format)

tab.to_csv("./data/results/ctab_vehicle_injury.csv")
