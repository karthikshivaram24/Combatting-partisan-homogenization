import pandas as pd
import os

if __name__ == "__main__":
    
    folder = "Files/Baseline_Results_MAPS/"
    files = [folder+f for f in os.listdir(folder) if ".csv" in f]
    
    for f in files:
        new_name = f.split(".")[0] +"_upd.csv"
        df = pd.read_csv(f)
        df = df.sort_values(by=['Baseline','Param_setting'], ascending=True)
        df.rename(columns={"avg_precision":"MAP","c1_avg_precision":"MAP_C1","c2_avg_precision":"MAP_C2"},inplace=True)
        print(df.columns)
        df = df[["Baseline","Param_setting","MAP","MAP_C1","MAP_C2"]]
        df = df.reset_index(drop=True)
        df.to_csv(new_name,index=False)