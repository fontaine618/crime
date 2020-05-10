import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

crashes = pd.read_csv("./data/raw/crashes.csv")

vehicules = pd.read_csv("./data/raw/vehicules.csv")
vehicules["CRASH_UNIT_ID"] = vehicules["CRASH_UNIT_ID"].astype("Int64")

people = pd.read_csv("./data/raw/people.csv")
people["VEHICLE_ID"] = people["VEHICLE_ID"].astype("Int64")

with open("./data/processed/car_on_car_both_occupied_crashes.txt", "r") as f:
    car_on_car = f.readlines()

car_on_car = "".join(car_on_car).replace("\n", ",").split(",")

# subset

crashes = crashes.loc[crashes["CRASH_RECORD_ID"].isin(car_on_car)]
# vehicules = vehicules.loc[vehicules["CRASH_RECORD_ID"].isin(car_on_car)]
# people = people.loc[people["CRASH_RECORD_ID"].isin(car_on_car)]
people = people.loc[people["PERSON_TYPE"].eq("DRIVER")]

# join drivers into their vehicle

vehicules_with_driver = vehicules.merge(
    right=people,
    how="inner",
    left_on="CRASH_UNIT_ID",
    right_on="VEHICLE_ID"
)

# join vehicules_with_drives into crashes

crashes_with_vehicules_and_driver = crashes.merge(
    right=vehicules_with_driver,
    how="inner",
    left_on="CRASH_RECORD_ID",
    right_on="CRASH_RECORD_ID_x"
)

good_records = crashes_with_vehicules_and_driver["CRASH_RECORD_ID_x"].value_counts().eq(2)
good_records = good_records.index[good_records].to_list()

df = crashes_with_vehicules_and_driver.loc[
    crashes_with_vehicules_and_driver["CRASH_RECORD_ID_x"].isin(good_records)
]

df.to_csv("./data/processed/crashes_two_cars_merged.csv", index=False)

# select columns

df["VEHICLE_TYPE"] = df["VEHICLE_TYPE"].replace({
    "PASSENGER": "PASSENGER",
    "SPORT UTILITY VEHICLE (SUV)": "SUV/PICKUP",
    "PICKUP": "SUV/PICKUP"
})

cols_to_keep =[
    'CRASH_RECORD_ID', 'CRASH_DATE',
       'POSTED_SPEED_LIMIT', 'TRAFFIC_CONTROL_DEVICE', 'DEVICE_CONDITION',
       'WEATHER_CONDITION', 'LIGHTING_CONDITION', 'FIRST_CRASH_TYPE',
       'TRAFFICWAY_TYPE', 'LANE_CNT', 'ALIGNMENT', 'ROADWAY_SURFACE_COND',
       'ROAD_DEFECT', 'CRASH_TYPE',
       'INTERSECTION_RELATED_I', 'NOT_RIGHT_OF_WAY_I', 'HIT_AND_RUN_I',
       'DAMAGE', 'PRIM_CONTRIBUTORY_CAUSE',
       'SEC_CONTRIBUTORY_CAUSE', 'STREET_DIRECTION',
       'BEAT_OF_OCCURRENCE', 'MOST_SEVERE_INJURY',
       'INJURIES_TOTAL', 'INJURIES_FATAL', 'INJURIES_INCAPACITATING',
       'INJURIES_NON_INCAPACITATING', 'INJURIES_REPORTED_NOT_EVIDENT',
       'INJURIES_NO_INDICATION', 'INJURIES_UNKNOWN', 'CRASH_HOUR',
       'CRASH_DAY_OF_WEEK', 'CRASH_MONTH', 'LATITUDE', 'LONGITUDE',
       'LOCATION',
    'CRASH_UNIT_ID',
       'VEHICLE_DEFECT', 'VEHICLE_TYPE',
       'TRAVEL_DIRECTION', 'MANEUVER',
       'OCCUPANT_CNT',
    'PERSON_ID', 'CITY', 'STATE',
       'ZIPCODE', 'SEX', 'AGE', 'DRIVERS_LICENSE_STATE',
       'DRIVERS_LICENSE_CLASS', 'SAFETY_EQUIPMENT', 'AIRBAG_DEPLOYED',
       'EJECTION', 'INJURY_CLASSIFICATION', 'DRIVER_ACTION', 'DRIVER_VISION',
       'BAC_RESULT', ]

df = df[cols_to_keep]

df.to_csv("./data/processed/crashes_two_cars_merged_columns_subset.csv", index=False)

# columns for crash

cols_crash = ['CRASH_RECORD_ID', 'CRASH_DATE', 'POSTED_SPEED_LIMIT',
       'TRAFFIC_CONTROL_DEVICE', 'DEVICE_CONDITION', 'WEATHER_CONDITION',
       'LIGHTING_CONDITION', 'FIRST_CRASH_TYPE', 'TRAFFICWAY_TYPE',
       'LANE_CNT', 'ALIGNMENT', 'ROADWAY_SURFACE_COND', 'ROAD_DEFECT',
       'CRASH_TYPE', 'INTERSECTION_RELATED_I', 'NOT_RIGHT_OF_WAY_I',
       'HIT_AND_RUN_I', 'DAMAGE', 'PRIM_CONTRIBUTORY_CAUSE',
       'SEC_CONTRIBUTORY_CAUSE', 'STREET_DIRECTION', 'BEAT_OF_OCCURRENCE',
       'MOST_SEVERE_INJURY', 'INJURIES_TOTAL', 'INJURIES_FATAL',
       'INJURIES_INCAPACITATING', 'INJURIES_NON_INCAPACITATING',
       'INJURIES_REPORTED_NOT_EVIDENT', 'INJURIES_NO_INDICATION',
       'INJURIES_UNKNOWN', 'CRASH_HOUR', 'CRASH_DAY_OF_WEEK',
       'CRASH_MONTH', 'LATITUDE', 'LONGITUDE', 'LOCATION']

cols_vehicle = [
       'CRASH_UNIT_ID',
       'VEHICLE_DEFECT', 'VEHICLE_TYPE', 'TRAVEL_DIRECTION', 'MANEUVER',
       'OCCUPANT_CNT']

cols_driver = ['PERSON_ID', 'SEX', 'AGE',
       'SAFETY_EQUIPMENT', 'AIRBAG_DEPLOYED', 'EJECTION',
       'INJURY_CLASSIFICATION', 'DRIVER_ACTION', 'DRIVER_VISION',
       'BAC_RESULT']

grouped_df = df.groupby("CRASH_RECORD_ID")

id, d = next(iter(grouped_df))


def vehicles_to_crash(d):
    crash = dict()
    # add common values
    for col in cols_crash:
        crash[col] = [d[col].iloc[0]]
    # add vehicules info
    for col in cols_vehicle + cols_driver:
        # randomize assignment to 0/1
        ids = np.random.choice(range(2), size=2, replace=False)
        for i, id in enumerate(ids):
            crash[col + "_" + str(i)] = [d[col].iloc[id]]
    return pd.DataFrame(crash)

# warning: takes a few minutes
crashes_wide = grouped_df.apply(vehicles_to_crash)

# Some checks on the new df

for _, col in crashes_wide.items():
    print(col.value_counts(dropna=False))

# POSTED_SPEED_LIMIT
limits = crashes_wide["POSTED_SPEED_LIMIT"].value_counts(dropna=False) > 50
limits = limits.index[limits].to_list()
crashes_wide = crashes_wide.loc[crashes_wide["POSTED_SPEED_LIMIT"].isin(limits)]
# LANE_CNT
limits = crashes_wide["LANE_CNT"].value_counts(dropna=False) > 50
limits = limits.index[limits].to_list()
crashes_wide = crashes_wide.loc[crashes_wide["LANE_CNT"].isin(limits)]
# VEHICLE_TYPE_0, VEHICLE_TYPE_1
types = ["PASSENGER", "SUV/PICKUP"]
crashes_wide["VEHICLE_TYPE_0"].value_counts(dropna=False)
crashes_wide = crashes_wide.loc[crashes_wide["VEHICLE_TYPE_0"].isin(types)]
crashes_wide["VEHICLE_TYPE_1"].value_counts(dropna=False)
crashes_wide = crashes_wide.loc[crashes_wide["VEHICLE_TYPE_1"].isin(types)]
# OCCUPANT_CNT_0, OCCUPANT_CNT_1
# ok after keeping good types
# AGE_0, AGE_1
limits = [*range(15, 100), np.nan]
crashes_wide = crashes_wide.loc[crashes_wide["AGE_0"].isin(limits)]
crashes_wide = crashes_wide.loc[crashes_wide["AGE_1"].isin(limits)]

crashes_wide.to_csv("./data/processed/crashes_wide.csv", index=False)