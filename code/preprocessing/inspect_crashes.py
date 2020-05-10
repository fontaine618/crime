import pandas as pd

crashes = pd.read_csv("./data/raw/crashes.csv")
log = pd.DataFrame(columns=["Condition", "n_True", "ids"])
todo = pd.Series(dtype=str)


# set index from CRASH_RECORD_ID
crashes.set_index("CRASH_RECORD_ID", inplace=True)

# RD_NO
crashes["RD_NO"].value_counts(dropna=False)

# CRASH_DATE_EST_I
crashes["CRASH_DATE_EST_I"].value_counts(dropna=False)
todo["CRASH_DATE_EST_I"] = "Lots of NaN"

# CRASH_DATE
crashes["CRASH_DATE"].value_counts(dropna=False)

# POSTED_SPEED_LIMIT
crashes["POSTED_SPEED_LIMIT"].value_counts(dropna=False)

# TRAFFIC_CONTROL_DEVICE
crashes["TRAFFIC_CONTROL_DEVICE"].value_counts(dropna=False)
todo["TRAFFIC_CONTROL_DEVICE"] = "Lots of small groups, maybe merge"

# DEVICE_CONDITION
crashes["DEVICE_CONDITION"].value_counts(dropna=False)
todo["DEVICE_CONDITION"] = "Lots of small groups, maybe merge"

# DEVICE_CONDITION
crashes["WEATHER_CONDITION"].value_counts(dropna=False)
todo["WEATHER_CONDITION"] = "Lots of small groups, maybe merge"

# LIGHTING_CONDITION
crashes["LIGHTING_CONDITION"].value_counts(dropna=False)

# FIRST_CRASH_TYPE
crashes["FIRST_CRASH_TYPE"].value_counts(dropna=False)
todo["FIRST_CRASH_TYPE"] = "A few small groups, maybe remove these cases"

# TRAFFICWAY_TYPE
crashes["TRAFFICWAY_TYPE"].value_counts(dropna=False)
todo["TRAFFICWAY_TYPE"] = "A few small groups, maybe remove these cases"

# LANE_CNT
crashes["LANE_CNT"].value_counts(dropna=False)

# ALIGNMENT
crashes["ALIGNMENT"].value_counts(dropna=False)
todo["ALIGNMENT"] = "A few small groups, maybe remove these cases"

# ROADWAY_SURFACE_COND
crashes["ROADWAY_SURFACE_COND"].value_counts(dropna=False)
todo["ROADWAY_SURFACE_COND"] = "A few small groups, maybe remove these cases"

# ROAD_DEFECT
crashes["ROAD_DEFECT"].value_counts(dropna=False)
todo["ROAD_DEFECT"] = "A few small groups, maybe remove these cases"

# REPORT_TYPE
crashes["REPORT_TYPE"].value_counts(dropna=False)
todo["REPORT_TYPE"] = "A few NaN"

# CRASH_TYPE
crashes["CRASH_TYPE"].value_counts(dropna=False)

# INTERSECTION_RELATED_I
crashes["INTERSECTION_RELATED_I"].value_counts(dropna=False)
todo["INTERSECTION_RELATED_I"] = "A lot of NaN"

# NOT_RIGHT_OF_WAY_I
crashes["NOT_RIGHT_OF_WAY_I"].value_counts(dropna=False)
todo["NOT_RIGHT_OF_WAY_I"] = "A lot of NaN"

# HIT_AND_RUN_I
crashes["HIT_AND_RUN_I"].value_counts(dropna=False)
todo["HIT_AND_RUN_I"] = "A lot of NaN"

# DAMAGE
crashes["DAMAGE"].value_counts(dropna=False)

# DATE_POLICE_NOTIFIED
crashes["DATE_POLICE_NOTIFIED"].value_counts(dropna=False)

# PRIM_CONTRIBUTORY_CAUSE
crashes["PRIM_CONTRIBUTORY_CAUSE"].value_counts(dropna=False)
todo["PRIM_CONTRIBUTORY_CAUSE"] = "A lot of cases, may need to merge"

# SEC_CONTRIBUTORY_CAUSE
crashes["SEC_CONTRIBUTORY_CAUSE"].value_counts(dropna=False)
todo["SEC_CONTRIBUTORY_CAUSE"] = "A lot of cases, may need to merge"

# STREET_NO
crashes["STREET_NO"].value_counts(dropna=False)

# STREET_DIRECTION
crashes["STREET_DIRECTION"].value_counts(dropna=False)
todo["STREET_DIRECTION"] = "A few NaN"

# STREET_NAME
crashes["STREET_NAME"].value_counts(dropna=False)
todo["STREET_NAME"] = "A few NaN"

# BEAT_OF_OCCURRENCE
crashes["BEAT_OF_OCCURRENCE"].value_counts(dropna=False)
todo["BEAT_OF_OCCURRENCE"] = "A few NaN"

# PHOTOS_TAKEN_I
crashes["PHOTOS_TAKEN_I"].value_counts(dropna=False)
todo["PHOTOS_TAKEN_I"] = "A lot of NaN"

# STATEMENTS_TAKEN_I
crashes["STATEMENTS_TAKEN_I"].value_counts(dropna=False)
todo["STATEMENTS_TAKEN_I"] = "A lot of NaN"

# DOORING_I
crashes["DOORING_I"].value_counts(dropna=False)
todo["DOORING_I"] = "A lot of NaN"

# WORK_ZONE_I
crashes["WORK_ZONE_I"].value_counts(dropna=False)
todo["WORK_ZONE_I"] = "A lot of NaN"

# WORK_ZONE_TYPE
crashes["WORK_ZONE_TYPE"].value_counts(dropna=False)
todo["WORK_ZONE_TYPE"] = "A lot of NaN"

# WORKERS_PRESENT_I
crashes["WORKERS_PRESENT_I"].value_counts(dropna=False)
todo["WORKERS_PRESENT_I"] = "A lot of NaN"

# NUM_UNITS
crashes["NUM_UNITS"].value_counts(dropna=False)
todo["NUM_UNITS"] = "A few NaN"

# MOST_SEVERE_INJURY
crashes["MOST_SEVERE_INJURY"].value_counts(dropna=False)
todo["MOST_SEVERE_INJURY"] = "A few NaN"

# INJURIES_TOTAL
crashes["INJURIES_TOTAL"].value_counts(dropna=False)
todo["INJURIES_TOTAL"] = "A few NaN"

# INJURIES_FATAL
crashes["INJURIES_FATAL"].value_counts(dropna=False)
todo["INJURIES_FATAL"] = "A few NaN"

# INJURIES_INCAPACITATING
crashes["INJURIES_INCAPACITATING"].value_counts(dropna=False)
todo["INJURIES_INCAPACITATING"] = "A few NaN"

# INJURIES_NON_INCAPACITATING
crashes["INJURIES_NON_INCAPACITATING"].value_counts(dropna=False)
todo["INJURIES_NON_INCAPACITATING"] = "A few NaN"

# INJURIES_REPORTED_NOT_EVIDENT
crashes["INJURIES_REPORTED_NOT_EVIDENT"].value_counts(dropna=False)
todo["INJURIES_REPORTED_NOT_EVIDENT"] = "A few NaN"

# INJURIES_NO_INDICATION
crashes["INJURIES_NO_INDICATION"].value_counts(dropna=False)
todo["INJURIES_NO_INDICATION"] = "A few NaN"

# INJURIES_UNKNOWN
crashes["INJURIES_UNKNOWN"].value_counts(dropna=False)
todo["INJURIES_UNKNOWN"] = "A few NaN or 0"

# CRASH_HOUR
crashes["CRASH_HOUR"].value_counts(dropna=False)

# CRASH_DAY_OF_WEEK
crashes["CRASH_DAY_OF_WEEK"].value_counts(dropna=False)
todo["CRASH_DAY_OF_WEEK"] = "Sunday=1"

# LATITUDE
crashes["LATITUDE"].value_counts(dropna=False)
todo["LATITUDE"] = "A few NaN"

# LONGITUDE
crashes["LONGITUDE"].value_counts(dropna=False)
todo["LONGITUDE"] = "A few NaN"

# LOCATION
crashes["LOCATION"].value_counts(dropna=False)
todo["LOCATION"] = "A few NaN"


# TRANSFORMATIONS

crashes = pd.read_csv("./data/raw/crashes.csv")
crashes.set_index("CRASH_RECORD_ID", inplace=True)

crashes["CRASH_DATE"] = pd.to_datetime(crashes["CRASH_DATE"])
before_2015 = (crashes["CRASH_DATE"].dt.year < 2015)
ids_before_2015 = crashes.index[before_2015].tolist()
log.loc["CRASH_DATE"] = ["<2015", len(ids_before_2015), ids_before_2015]

crashes["LANE_CNT"] = crashes["LANE_CNT"].astype('Int64')

crashes["DATE_POLICE_NOTIFIED"] = pd.to_datetime(crashes["DATE_POLICE_NOTIFIED"])
before_2015 = (crashes["DATE_POLICE_NOTIFIED"].dt.year < 2015)
ids_before_2015 = crashes.index[before_2015].tolist()
log.loc["DATE_POLICE_NOTIFIED"] = ["<2015", len(ids_before_2015), ids_before_2015]

crashes["NUM_UNITS"] = crashes["NUM_UNITS"].astype('Int64')
crashes["INJURIES_TOTAL"] = crashes["INJURIES_TOTAL"].astype('Int64')
crashes["BEAT_OF_OCCURRENCE"] = crashes["BEAT_OF_OCCURRENCE"].astype('Int64')

crashes["INJURIES_FATAL"] = crashes["INJURIES_FATAL"].astype('Int64')
crashes["INJURIES_INCAPACITATING"] = crashes["INJURIES_INCAPACITATING"].astype('Int64')
crashes["INJURIES_NON_INCAPACITATING"] = crashes["INJURIES_NON_INCAPACITATING"].astype('Int64')
crashes["INJURIES_REPORTED_NOT_EVIDENT"] = crashes["INJURIES_REPORTED_NOT_EVIDENT"].astype('Int64')
crashes["INJURIES_NO_INDICATION"] = crashes["INJURIES_NO_INDICATION"].astype('Int64')
crashes["INJURIES_UNKNOWN"] = crashes["INJURIES_UNKNOWN"].astype('Int64')
