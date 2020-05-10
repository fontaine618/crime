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
pd.set_option('display.width', 1000)

# ============== Import ===================
crashes = pd.read_csv("./data/processed/crashes_final.csv")
crashes.set_index("CRASH_RECORD_ID", inplace=True)


# ============== Setup ===================
features = [
    'PRIM_CONTRIBUTORY_CAUSE',
    'CRASH_MONTH', 'CRASH_SEASON',
    'CRASH_DAY_OF_WEEK', 'CRASH_WEEKDAY',
    'CRASH_HOUR', 'CRASH_DARK',
    'VEHICLE_TYPE_0', 'VEHICLE_TYPE_1',
    'MANEUVER_0', 'MANEUVER_1',
    'SEX_0', 'SEX_1',
    'AGE_BINNED_0', 'AGE_BINNED_1',
    'SAFETY_EQUIPMENT_0', 'SAFETY_EQUIPMENT_1',
    'DRIVER_ACTION_0', 'DRIVER_ACTION_1',
    'BAC_RESULT_0', 'BAC_RESULT_1'
]

features_main = [
    'PRIM_CONTRIBUTORY_CAUSE',
    'MANEUVER_0', 'MANEUVER_1',
    'SEX_0', 'SEX_1',
    'SAFETY_EQUIPMENT_0', 'SAFETY_EQUIPMENT_1',
    'AGE_BINNED_0', 'AGE_BINNED_1',
    'DRIVER_ACTION_0', 'DRIVER_ACTION_1',
    'BAC_RESULT_0', 'BAC_RESULT_1'
]
formula_main = " + ".join(features_main)

responses = ["INJURIES_0_ANY", "INJURIES_0_SEVERE"]
months = ["GAM", "BINNED"]
weekdays = ["ALL", "BINNED"]
hours = ["GAM", "BINNED"]



def formula(response, month, weekday, hour):
    f = response + " ~ "
    f += "VEHICLE_TYPE_0 * VEHICLE_TYPE_1"

    f += " + " + formula_main
    gam = []

    if month == "GAM":
        gam.append("CRASH_MONTH")
    elif month == "BINNED":
        f += " + CRASH_SEASON"

    if weekday == "ALL":
        f += " + CRASH_DAY_OF_WEEK"
    elif weekday == "BINNED":
        f += " + CRASH_WEEKDAY"

    if hour == "GAM":
        gam.append("CRASH_HOUR")
    elif hour == "BINNED":
        f += " + CRASH_DARK"


    spline = None
    if len(gam) > 0:
        spline = BSplines(crashes[gam], df=[5]*len(gam), degree=[3]*len(gam))

    return f, spline



# =============== Run models ===================
results = pd.DataFrame(columns=[
    "Family", "Formula", "RESPONSE", "MONTH", "WEEKDAY", "HOUR",
    "LogLik", "Deviance", "Df_Model", "AIC", "BIC", "model", "fitted_model"
])

experiments = itertools.product(responses, months, weekdays, hours)
for k, (response, month, weekday, hour) in enumerate(experiments):
    print(k, response, month, weekday, hour)
    f, s = formula(response, month, weekday, hour)
    if s is not None:
        model = GLMGam.from_formula(
            formula=f,
            smoother=s,
            data=crashes,
            family=sm.families.Binomial()
        )
    else:
        model = sm.GLM.from_formula(
            formula=f,
            data=crashes,
            family=sm.families.Binomial()
        )
    res = model.fit()
    metrics = (res.llf, res.deviance, res.df_model, res.aic, res.bic)
    print(metrics)
    results.loc[k] = (
        "Binomial", f,
        response, month, weekday, hour,
        *metrics, model, res
    )

results.drop(columns=["Formula", "Family", "model", "fitted_model"])

results.drop(columns=["Formula", "Family", "model", "fitted_model"]).to_csv("./data/results/logistic_regression.csv")


best_any = 7
model_any = results.loc[best_any, "fitted_model"]
best_severe = 15
model_severe = results.loc[best_severe, "fitted_model"]

models = {
    "INJURIES_0_ANY": model_any,
    "INJURIES_0_SEVERE": model_severe
}

coef_table = coef_table_wide(models, columns=["coef", "se", "p", "ci"])
coef_table.to_csv("./data/results/coef_table.csv")


# # ================= GAM results ===============
#
# model_any_ages_0 = model_any.partial_values(0)
# model_any_ages_1 = model_any.partial_values(1)
# model_any_months = model_any.partial_values(2)
#
#
# model_severe_ages_0 = model_severe.partial_values(0)
# model_severe_ages_1 = model_severe.partial_values(1)
#
# gam_preds = crashes[["AGE_0", "AGE_1", "CRASH_MONTH"]]
#
# gam_preds["ANY_AGE_0_MEAN"] = model_any_ages_0[0]
# gam_preds["ANY_AGE_0_SE"] = model_any_ages_0[1]
# gam_preds["ANY_AGE_1_MEAN"] = model_any_ages_1[0]
# gam_preds["ANY_AGE_1_SE"] = model_any_ages_1[1]
# gam_preds["ANY_MONTH_MEAN"] = model_any_months[0]
# gam_preds["ANY_MONTH_SE"] = model_any_months[1]
#
# gam_preds["SEVERE_AGE_0_MEAN"] = model_severe_ages_0[0]
# gam_preds["SEVEREY_AGE_0_SE"] = model_severe_ages_0[1]
# gam_preds["SEVERE_AGE_1_MEAN"] = model_severe_ages_1[0]
# gam_preds["SEVERE_AGE_1_SE"] = model_severe_ages_1[1]
#
# gam_preds.to_csv("./data/results/gam_preds.csv")

# ================= Contrasts ===============

p = len(model_any.params)
interaction = list(model_any.params.index.values).index(
    "VEHICLE_TYPE_0[T.SUV/PICKUP]:VEHICLE_TYPE_1[T.SUV/PICKUP]"
)

contrasts = np.zeros((4, p))
contrasts[:, 0] = 0
contrasts[1, 1] = 1
contrasts[2, 2] = 1
contrasts[3, 1] = 1
contrasts[3, 2] = 1
contrasts[3, interaction] = 1

ttest_any = model_any.t_test(contrasts).summary_frame()

ttest_any["Response"] = "INJURIES_0_ANY"


p = len(model_severe.params)
interaction = list(model_severe.params.index.values).index(
    "VEHICLE_TYPE_0[T.SUV/PICKUP]:VEHICLE_TYPE_1[T.SUV/PICKUP]"
)

contrasts = np.zeros((4, p))
contrasts[:, 0] = 0
contrasts[1, 1] = 1
contrasts[2, 2] = 1
contrasts[3, 1] = 1
contrasts[3, 2] = 1
contrasts[3, interaction] = 1

ttest_severe = model_severe.t_test(contrasts).summary_frame()

ttest_severe["Response"] = "INJURIES_0_SEVERE"



ttests = pd.concat([ttest_any, ttest_severe])

ttests["VEHICLE_TYPE_0"] = [
    "PASSENGER", "SUV/PICKUP"
] * 4
ttests["VEHICLE_TYPE_1"] = [
    "PASSENGER", "PASSENGER", "SUV/PICKUP", "SUV/PICKUP"
] * 2

ttests.to_csv("./data/results/contrasts.csv")