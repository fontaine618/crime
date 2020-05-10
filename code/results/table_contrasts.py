import pandas as pd
import sys
sys.path.append("/home/simon/Documents/sithon")
from sithon.latex.pd_to_latex import pd_to_tabular

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width', 1000)


tab = pd.read_csv("./data/results/contrasts.csv", index_col=0)

tab.set_index(["VEHICLE_TYPE_0", "VEHICLE_TYPE_1"], inplace=True)
groups = tab["Response"]
tab.drop(columns=["Response"], inplace=True)

tab = tab.applymap("{:.03f}".format)

tab = tab.groupby(groups)

table = pd_to_tabular(
    tab,
    title="Logistic Regression Vehicule Type Contrasts",
    column_format="llrrrrrr"
)

print(table)

with open("./report/tables/contrasts.tex", "w") as f:
    f.write(table)
