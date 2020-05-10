import pandas as pd
import sys
sys.path.append("/home/simon/Documents/sithon")
from sithon.latex.pd_to_latex import pd_to_tabular

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width', 1000)


tab = pd.read_csv("./data/results/coef_table.csv", header=[0, 1], index_col=0)

var = []
val = []

for v in tab.index:
    sq = v.find('[')
    if sq > 0:
        var.append(v[:sq])
        val.append(v[(sq+3):-1])
    else:
        var.append(v)
        val.append("")

tab.index = val
tab = tab.iloc[:, [0, 1, 5, 6]]


tab = tab.applymap("{:.02f}".format)

for i, (v, va) in enumerate(zip(val, var)):
    print(i, va, v)


# CRASH
rows = [*range(3, 10), *range(58, 63)]
vars = [var[row] for row in rows]
tab_gr = tab.iloc[rows].groupby(vars)
table = pd_to_tabular(
    tab_gr,
    title="Logistic Regression: Crash Control Variates Estimates",
    column_format="lrrrr",
    group_name_align="l",
    escape=False
)
table = table.replace("_", "\_")
table = table.replace("%", "\%")
print(table)

with open("./report/tables/control_variates_crash.tex", "w") as f:
    f.write(table)


# VEHICLE
rows = [*range(10, 24), *range(38, 56)]
vars = [var[row] for row in rows]
tab_gr = tab.iloc[rows].groupby(vars)
table = pd_to_tabular(
    tab_gr,
    title="Logistic Regression: Vehicle Control Variates Estimates",
    column_format="lrrrr",
    group_name_align="l",
    escape=False
)
table = table.replace("_", "\_")
table = table.replace("%", "\%")
print(table)

with open("./report/tables/control_variates_vehicle.tex", "w") as f:
    f.write(table)



# DRIVER
rows = [*range(24, 38), *range(56, 58)]
vars = [var[row] for row in rows]
tab_gr = tab.iloc[rows].groupby(vars)
table = pd_to_tabular(
    tab_gr,
    title="Logistic Regression: Driver Control Variates Estimates",
    column_format="lrrrr",
    group_name_align="l",
    escape=False
)
table = table.replace("_", "\_")
table = table.replace("%", "\%")
print(table)

with open("./report/tables/control_variates_driver.tex", "w") as f:
    f.write(table)
