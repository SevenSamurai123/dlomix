import os
import tensorflow as tf
import pathlib
import copy

import dlomix
from dlomix.dlomix.data import FragmentIonIntensityDataset
from dlomix.dlomix.losses import masked_spectral_distance
from dlomix.dlomix.models import PrositIntensityPredictor

from PTM.scripts.utility import utility
from PTM.scripts.mod_model_weights import handle_weights

class CleverCopyPredictorBeforeFineTuning:

    """
        Initialize the CleverCopyPredictor for Fine-tuning.
        Args:
            modification (str): The modification e.g. R[UNIMOD:7].
            train_path (str): Path to the training dataset.
            test_path (str): Path to the test dataset.
            val_path (str): Path to the validation dataset.
            path_weights (str): Path to the model weights used for traning the model.
            save_path (str): Path to save the results.
            amino_acid (str): The amino acid to be modified.
            init_path (str, optional): Path for initialize/evaluate the model, to be able to load weights. Defaults to None.
            case (str, optional): Case type for handling different scenarios. Normal: Normal Clever Copy withput chaining.
                                Chaining_start: Start modification of chaining. Chaining: Modification in chain, not at start.
                                Unseen: Predict unseen modification.
            mods_to_add_to_alphabet (list, optional): List of modifications to add to the alphabet. Only needed for chaining. Defaults to None.
            remove_unnecessary_weights (bool, optional): Delete all weight files of te current run, except of the best AA. Defaults to None.
            save_output_to_file (bool, optional): Flag to save output to a file. Defaults to False.
    """
    def __init__(self, modification, test_path, val_path, path_weights, save_path,init_path=None, case=None, mods_to_add_to_alphabet=None, remove_unnecessary_weights=False, save_output_to_file=False):
        self.modification = modification
        self.test_path = test_path
        self.val_path = val_path
        self.init_path = init_path
        self.path_weights = path_weights
        self.save_path = save_path
        self.case = case
        self.mods_to_add_to_alphabet = mods_to_add_to_alphabet
        self.remove_unnecessary_weights = remove_unnecessary_weights
        self.save_output_to_file = save_output_to_file

        if self.init_path==None:
            self.init_path = self.val_path

        self.modified_AA = self.modification[-1]
        #self.shortcut_mod = utility.get_shortcut_of_mod(self.modification)
        self.tmp_weights = "tmp.weights.h5"
        
        self.alphabet_AA = copy.deepcopy(dlomix.dlomix.constants.ALPHABET_UNMOD)
        self.dict_AA_SA = {}

        self.config = {
            "batch_size": 512,
            "lr": 0.0001
        }

        self.setup_paths()
        self.setup_alphabet()
        self.create_datasets()
        self.create_model()

    def setup_paths(self):
        """
        Setup the directory paths for saving results, weights, and validation/test outputs.
        """
        print("setup paths")

        if self.case != "unseen":
            self.save_final_results = self.save_path
            self.save_path_before = f'{self.save_path}/{self.modification}/before'
        else:
            self.save_final_results = self.save_path
            self.save_path_before = f'{self.save_path}/unseen/{self.modification}/before'
        
        self.save_path_test = f'{self.save_path_before}/test'
        self.save_path_val = f'{self.save_path_before}/validation'
        self.save_path_weights = f'{self.save_path_before}/weights'
        
        # Create necessary directories
        pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.save_path_weights).mkdir(parents=True, exist_ok=True)

    def setup_alphabet(self):
        """
        Add new modification(s) to the alphabet.
        """
        print("setup alphabet")

        if self.case == "chaining" or self.case == "unseen":
            for mod in self.mods_to_add_to_alphabet:
                unimod_new = utility.get_modification_in_UNIMOD(modification=mod)
                self.alphabet_AA[unimod_new] = len(self.alphabet_AA) + 1
                #self.alphabet_AA[mod] = len(self.alphabet_AA) + 1
            unimod_new = utility.get_modification_in_UNIMOD(modification=self.modification)
            self.alphabet_AA[unimod_new] = len(self.alphabet_AA) + 1
            #self.alphabet_AA[self.modification] = len(self.alphabet_AA) + 1

        elif self.case == "normal" or self.case=="chaining_start" or self.case==None:
            unimod = utility.get_modification_in_UNIMOD(modification=self.modification)
            self.alphabet_AA[unimod] = len(self.alphabet_AA) + 1
            #self.alphabet_AA[self.modification] = len(self.alphabet_AA) + 1

        print(self.alphabet_AA)
          
    def create_datasets(self):
        """
        Create datasets for training, validation, and testing using FragmentIonIntensityDataset.
        """
        print("create datasets")

        # Create datasets
        self.validation = FragmentIonIntensityDataset(
            data_format="parquet",
            sequence_column="modified_sequence",
            label_column="intensities_raw",
            model_features=["precursor_charge_onehot", "collision_energy_aligned_normed"],
            max_seq_len=30,
            batch_size=self.config["batch_size"],
            alphabet=self.alphabet_AA,
            test_data_source=self.val_path,
            encoding_scheme="naive-mods",
            with_termini=False
        )

        self.test = FragmentIonIntensityDataset(
            data_format="parquet",
            sequence_column="modified_sequence",
            label_column="intensities_raw",
            model_features=["precursor_charge_onehot", "collision_energy_aligned_normed"],
            max_seq_len=30,
            batch_size=self.config["batch_size"],
            test_data_source=self.test_path,
            alphabet=self.alphabet_AA,
            encoding_scheme="naive-mods",
            with_termini=False
        )

        self.init_model = FragmentIonIntensityDataset(
            data_format="parquet",
            sequence_column="modified_sequence",
            label_column="intensities_raw",
            model_features=["precursor_charge_onehot", "collision_energy_aligned_normed"],
            max_seq_len=30,
            batch_size=self.config["batch_size"],
            test_data_source=self.init_path,
            alphabet=self.alphabet_AA,
            encoding_scheme="naive-mods",
            with_termini=False
        )

    def create_model(self):
        """
        Create and compile the PrositIntensityPredictor model for training.
        """
        print("create model")

        self.model = PrositIntensityPredictor(
            seq_length=30,
            input_keys={
                "SEQUENCE_KEY": "modified_sequence",
                "COLLISION_ENERGY_KEY": "collision_energy_aligned_normed",
                "PRECURSOR_CHARGE_KEY": "precursor_charge_onehot",
            },
            meta_data_keys={ 
                "COLLISION_ENERGY_KEY": "collision_energy_aligned_normed",
                "PRECURSOR_CHARGE_KEY": "precursor_charge_onehot",
            },
            with_termini=False,
            alphabet=self.alphabet_AA
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.model.compile(optimizer=optimizer, loss=masked_spectral_distance, metrics=["mse"])

    def prediction(self):
        """
        Predict the ion fragment intensities and calculate the spectral angle .
        """
        print("prediction before fine-tuning")

        self.model.evaluate(self.init_model.tensor_test_data,verbose=0)

        weights = handle_weights(hpf5_file=self.path_weights, save_path=self.tmp_weights)
        weights.save_custom_weights_to_file(model=self.model, numNewRows=len(self.alphabet_AA)-(len(self.alphabet_AA)-1))
        self.model.load_weights(self.tmp_weights)

        embedding_weights = self.model.layers[0].get_weights()[0]

        alphabet_keys = list(self.alphabet_AA.keys())
        for AA, row in zip(alphabet_keys, embedding_weights[1:-1]):   
            weights = embedding_weights
            weights[-1] = row
            self.model.layers[0].set_weights([weights])

            self.model.save_weights(f"{self.save_path_weights}/{self.modification}_{AA}_max_before.weights.h5")

            predictions_val = self.model.predict(self.validation.tensor_test_data,verbose=0)
            df_spectral_angle_val = utility.get_spectral_angle(self.validation, predictions_val)
            spectral_angle_mean_val = df_spectral_angle_val['mean']
            #print(f"Val, Mod: {self.modification}, AA: {AA}, SA: {spectral_angle_mean_val}")

            predictions_test = self.model.predict(self.test.tensor_test_data,verbose=0)
            df_spectral_angle_test = utility.get_spectral_angle(self.test, predictions_test)
            spectral_angle_mean_test = df_spectral_angle_test['mean']
            #print(f"Test, Mod: {self.modification}, AA: {AA}, SA: {spectral_angle_mean_test}")

            self.dict_AA_SA[AA] = spectral_angle_mean_test

            if self.save_output_to_file:
                pathlib.Path(self.save_path_test).mkdir(parents=True, exist_ok=True)
                pathlib.Path(self.save_path_val).mkdir(parents=True, exist_ok=True)
                utility.write_spectral_angle_to_file(self.save_path_val, self.modification, "max", df_spectral_angle_val, AA, "before")
                utility.write_spectral_angle_to_file(self.save_path_test, self.modification, "max", df_spectral_angle_test, AA, "before")

        self.dict_AA_SA = dict(sorted(self.dict_AA_SA.items(), key=lambda item: item[1],reverse=True))
        self.best_amino_acid = list(self.dict_AA_SA.keys())[0]
        
        if self.remove_unnecessary_weights:
            self.cleanup_weights()

    def cleanup_weights(self):
        for filename in os.listdir(self.save_path_weights):
            search_string = f"_{self.best_amino_acid}_" 
            if search_string not in filename:
                file_path = os.path.join(self.save_path_weights, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    def get_AA_SA_dict(self):
        return self.dict_AA_SA

    def get_SA_of_AA(self,amino_acid):
        return self.dict_AA_SA[amino_acid]

    def get_best_AA(self):
        return self.best_amino_acid
    
    def get_weights_path_of_best_AA(self):
        for filename in os.listdir(self.save_path_weights):
            if self.best_amino_acid in filename:
                self.file_path = os.path.join(self.save_path_weights, filename)
        return self.file_path
    
    def get_SA_before_ft(self):
        return self.dict_AA_SA[self.best_amino_acid]
