import pandas as pd
import re
import itertools
import pathlib

from PTM.scripts.utility import utility

class DataPreprocessing:

    def __init__(self,file_path,save_path,modification,train_ratio, fragmentation="HCD", mass_analyzer="FTMS", allowed_modifications=None,create_smaller_trainset=None):
        self.file_path=str(file_path)
        self.save_path = str(save_path)
        self.modification = str(modification)
        self.train_ratio = float(train_ratio)
        self.fragmentation = str(fragmentation)
        self.mass_analyzer = str(mass_analyzer)
        self.allowed_modifications = allowed_modifications
        self.create_smaller_trainsets = create_smaller_trainset

        self.shortcut_mod = utility.get_shortcut_of_mod(self.modification)

        self.df = pd.read_parquet(self.file_path)

        if self.allowed_modifications==None:
            self.allowed_modifications = []
            self.allowed_modifications.append(self.modification)
        else:
            self.allowed_modifications = list(allowed_modifications)
            self.allowed_modifications.append(self.modification)

        pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)

        self.preprocess_data()
        self.create_train_test_val()

        if self.create_smaller_trainsets!=None:
            self.create_smaller_trainsets = list(self.create_smaller_trainsets)
            self.create_smaller_trainsets_of_size_N(f"{self.save_path}/train_{self.fragmentation}_{self.shortcut_mod}_max.parquet")


    def preprocess_data(self):
        
        self.df = self.df[(self.df['fragmentation']==self.fragmentation) & (self.df['mass_analyzer']==self.mass_analyzer)]

        self.df = self.df[['modified_sequence','intensities_raw','collision_energy_aligned_normed','precursor_charge_onehot']]
        
        processed_modification = re.sub(r'(\[|\])', r'\\\1', self.modification)
        self.df = self.df[self.df['modified_sequence'].str.contains(processed_modification)]
        
        self.df['modified_sequence'] = self.df['modified_sequence'].apply(lambda x: self.remove_terminals(x))

        mods_in_data = self.find_all_mods(self.df)
        for mod in mods_in_data:
            if mod not in self.allowed_modifications:
                processed_mod = re.sub(r'(\[|\])', r'\\\1', mod)
                self.df = self.df[~self.df['modified_sequence'].str.contains(processed_mod)]
        
        self.processed_df = self.df

        return self.processed_df


    def create_train_test_val(self):
        self.train, self.test, self.val = self.create_sets(self.df)

        self.train.to_parquet(f"{self.save_path}/train_{self.fragmentation}_{self.shortcut_mod}_max.parquet")
        self.test.to_parquet(f"{self.save_path}/test_{self.fragmentation}_{self.shortcut_mod}_max.parquet")
        self.val.to_parquet(f"{self.save_path}/val_{self.fragmentation}_{self.shortcut_mod}_max.parquet")

    def find_all_mods(self,df):
        a = [re.findall(r'\[\]-|-\[\]|[A-Z]?\[UNIMOD:\d+\]', str(x)) for x in df['modified_sequence']]
        b = [x for x in a if x != []]
        c = list(itertools.chain(*b))
        return list(set(c))
    
    def create_sets(self,data):
        train = data
        train = train[0:0]
        
        test = data
        test = test[0:0]
        
        val = data
        val = val[0:0]

        counts = pd.DataFrame(data["modified_sequence"].value_counts())
        train_table = counts.sample(frac=self.train_ratio, random_state=42)
        df_rest = counts.loc[~counts.index.isin(train_table.index)]
        test_table = df_rest.sample(frac=0.5, random_state=42)
        val_table = df_rest.loc[~df_rest.index.isin(test_table.index)]

        for sequence, count in train_table.iterrows():
            a = data.loc[data['modified_sequence']==sequence]
            train = pd.concat([train, pd.DataFrame(a)])

        for sequence, count in test_table.iterrows():
            a = data.loc[data['modified_sequence']==sequence]
            test = pd.concat([test, pd.DataFrame(a)])

        for sequence, count in val_table.iterrows():
            a = data.loc[data['modified_sequence']==sequence]
            val = pd.concat([val, pd.DataFrame(a)])

        train.reset_index(inplace=True)
        test.reset_index(inplace=True)
        val.reset_index(inplace=True)

        return train, test, val
    
    def create_smaller_trainsets_of_size_N(self,path):
        train_data = pd.read_parquet(path)
        for size in self.create_smaller_trainsets:
            if len(train_data)>size:
                train_table = train_data.sample(n = size, random_state=42, axis=0, replace=False)
                train_table.to_parquet(f"{self.save_path}/train_{self.fragmentation}_{self.shortcut_mod}_{size}.parquet")
                print(f"Create train set of size: {size}")
            else:
                print(f"Trainset size is {len(train_data)}. Cannot create subset of size {size}.")
    
    def remove_terminals(self,s):
        replace_pattern = r'\[\]-|-\[\]'
        cleaned_string = re.sub(replace_pattern, '', s)
        return cleaned_string
    
    def get_train_set(self):
        return self.train
    
    def get_test_set(self):
        return self.test
    
    def get_val_set(self):
        return self.val

    def get_processed_input_data(self):
        return self.processed_df