import pandas as pd

crashes = pd.read_csv("./data/raw/crashes.csv")
crashes.set_index("CRASH_RECORD_ID", inplace=True)
vehicules = pd.read_csv("./data/raw/vehicules.csv")
vehicules.set_index("CRASH_UNIT_ID", inplace=True)
people = pd.read_csv("./data/raw/people.csv")
people.set_index("PERSON_ID", inplace=True)

# We identify crashes involving exactly two vehicules

crashes["NUM_UNITS"] = crashes["NUM_UNITS"].astype('Int64')
crashes["NUM_UNITS"].value_counts(dropna=False)
crashes_id_2 = crashes["NUM_UNITS"].eq(2)
crashes_id_2.fillna(False, inplace=True)
crashes_id_2 = crashes.index[crashes_id_2].to_list()

# Subset vehicules

vehicules = vehicules.loc[vehicules["CRASH_RECORD_ID"].isin(crashes_id_2)]

# Find cars

vehicules["VEHICLE_TYPE"].unique()

cars = ['PASSENGER', 'SPORT UTILITY VEHICLE (SUV)', 'PICKUP']

vehicules = vehicules.loc[vehicules["VEHICLE_TYPE"].isin(cars)]

# Find occupied vehicules

occupied = vehicules["OCCUPANT_CNT"] > 0

occupied = occupied.index[occupied]

vehicules = vehicules.loc[occupied]

# Has driver info

people = people.loc[people["VEHICLE_ID"].isin(occupied)]
people["PERSON_TYPE"].value_counts()
people = people.loc[people["PERSON_TYPE"].eq("DRIVER")]

# Find crashes with exactly two instances

car_on_car = people["CRASH_RECORD_ID"].value_counts().eq(2)
car_on_car = car_on_car.index[car_on_car].to_list()

with open("./data/processed/car_on_car_both_occupied_crashes.txt", "w") as f:
    f.write("\n".join(car_on_car))