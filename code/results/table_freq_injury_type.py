import pandas as pd
import sys
sys.path.append("/home/simon/Documents/sithon")
from sithon.latex.pd_to_latex import pd_to_tabular

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width', 1000)


tab = pd.read_csv("./data/results/ctab_vehicle_injury.csv",
                  header=[0, 1, 2], index_col=0)

table = pd_to_tabular(
    tab,
    title="Injury Frequency by Vehicle Type",
    column_format="lrrrrrrrr"
)

print(table)

with open("./report/tables/injury_frequency.tex", "w") as f:
    f.write(table)
