import pandas as pd

people = pd.read_csv("./data/raw/people.csv")

for col in people.columns:
    print(people[col].value_counts(dropna=False))

print(people.columns)

# TRANSFORMATIONS

people.set_index("PERSON_ID", inplace=True)

people["CRASH_DATE"] = pd.to_datetime(people["CRASH_DATE"])
people["VEHICLE_ID"] = people["VEHICLE_ID"].astype('Int64')
people["SEAT_NO"] = people["SEAT_NO"].astype('Int64')
people["VEHICLE_ID"] = people["VEHICLE_ID"].astype('Int64')
people["VEHICLE_ID"] = people["VEHICLE_ID"].astype('Int64')
people["VEHICLE_ID"] = people["VEHICLE_ID"].astype('Int64')
people["VEHICLE_ID"] = people["VEHICLE_ID"].astype('Int64')
people["VEHICLE_ID"] = people["VEHICLE_ID"].astype('Int64')
people["VEHICLE_ID"] = people["VEHICLE_ID"].astype('Int64')
people["VEHICLE_ID"] = people["VEHICLE_ID"].astype('Int64')
people["VEHICLE_ID"] = people["VEHICLE_ID"].astype('Int64')
