import pandas as pd
import sys
sys.path.append("/home/simon/Documents/sithon")
from sithon.latex.pd_to_latex import pd_to_tabular

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width', 1000)


tab = pd.read_csv("./data/results/logistic_regression.csv", index_col=0)

tab.set_index(["MONTH", "WEEKDAY", "HOUR"], inplace=True)
groups = tab["RESPONSE"]
tab.drop(columns=["Deviance", "RESPONSE"], inplace=True)

tab = tab.applymap("{:.0f}".format)

tab = tab.groupby(groups)

table = pd_to_tabular(
    tab,
    title="Logistic Regression Models Results",
    column_format="lllrrrr"
)

print(table)

with open("./report/tables/model_selection.tex", "w") as f:
    f.write(table)
