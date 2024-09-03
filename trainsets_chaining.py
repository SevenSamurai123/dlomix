import pandas as pd
import pathlib
import os

from PTM.scripts.utility import utility


class Trainsets_chaining:

    def __init__(self,chain, input_path_files,save_path,trainsize="max", fragmentation = "HCD",factor=2):
        self.chain = chain
        self.input_path_files = input_path_files
        self.save_path=save_path
        self.trainsize = trainsize
        self.fragmentation = fragmentation
        self.factor = float(factor)

        pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)

        self.create_chain_train_data()

    def create_chain_train_data(self):
        start_mod = self.chain[0]
        save_name_all = "-".join(self.chain)

        for i, mod in enumerate(self.chain):
            save_string = ""
            final_df = pd.DataFrame() 

            if os.path.exists(f"{self.input_path_files}/train_{self.fragmentation}_{mod}_{self.trainsize}.parquet"):
            #if os.path.exists(f"/cmnfs/home/students/c.kloppert/dlomix_hugginface/example/train_HCD_R[UNIMOD:7]_max.parquet"):
                df_current = pd.read_parquet(f"{self.input_path_files}/train_{self.fragmentation}_{mod}_{self.trainsize}.parquet")
                save_string = f'{len(df_current)} {mod} + '
            else:
                df_current = pd.read_parquet(f"{self.input_path_files}/train_{self.fragmentation}_{mod}_max.parquet")
                save_string = f'{len(df_current)} {mod} + '

            if start_mod != mod:
                final_df = pd.concat([final_df, df_current])

                for j, _mod in enumerate(self.chain[0:i]):
                    
                    if os.path.exists(f"{self.input_path_files}/train_{self.fragmentation}_{_mod}_{self.trainsize}.parquet"):
                        df_previous = pd.read_parquet(f"{self.input_path_files}/train_{self.fragmentation}_{_mod}_{self.trainsize}.parquet")
                    else:
                        df_previous = pd.read_parquet(f"{self.input_path_files}/train_{self.fragmentation}_{_mod}_max.parquet")
    
                    sample_size = int(len(df_previous) / self.factor)

                    train_data_sample = df_previous.sample(n=sample_size, random_state=42, replace=False)
                    save_string += f'{len(train_data_sample)} {_mod} + '
                    final_df = pd.concat([final_df, train_data_sample])

                final_df.drop('index', axis=1, inplace=True, errors='ignore')
                final_df = final_df.sample(frac=1).reset_index(drop=True)

            save_string = save_string.strip(' + ')  
            print(f"{mod}: {save_string}")

            save_name = "-".join(self.chain[:i+1])
            output_dir = f"{self.save_path}/{save_name_all}"
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

            output_path = f"{output_dir}/train_{self.fragmentation}_{mod}_{save_name}_{self.trainsize}.parquet"
            final_df.to_parquet(output_path)
            

