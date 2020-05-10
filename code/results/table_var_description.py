import pandas as pd
import sys
sys.path.append("/home/simon/Documents/sithon")
from sithon.latex.pd_to_latex import pd_to_tabular

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width', 1000)

vars_crash = pd.read_csv("./data/results/vars_crash.csv", index_col=0).fillna("")
vars_vehicle = pd.read_csv("./data/results/vars_vehicle.csv", index_col=0).fillna("")
vars_drivers = pd.read_csv("./data/results/vars_drivers.csv", index_col=0).fillna("")

vars = pd.concat([vars_crash, vars_vehicle, vars_drivers], 0)
vars.index.name = "Variable"

groups = ["Unit", "Calculated"]


vars_grouped = vars.drop(columns=groups + ["Description"]).groupby([vars[gr] for gr in groups])


table = pd_to_tabular(
    vars_grouped,
    title="Variable Description",
    column_format="lll",
    escape=False
)

table = table.replace("_", "\\_")
print(table)

with open("./report/tables/var_decription.tex", "w") as f:
    f.write(table)
