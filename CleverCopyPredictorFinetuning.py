import tensorflow as tf
import pathlib
import copy

import dlomix
from dlomix.dlomix.data import FragmentIonIntensityDataset
from dlomix.dlomix.losses import masked_spectral_distance
from dlomix.dlomix.models import PrositIntensityPredictor

from PTM.scripts.utility import utility

class CleverCopyPredictorFineTuning:
    """
        Initialize the CleverCopyPredictor for Fine-tuning.
        Args:
            modification (str): The modification e.g. R[UNIMOD:7].
            train_path (str): Path to the training dataset.
            test_path (str): Path to the test dataset.
            val_path (str): Path to the validation dataset.
            trainsize (int): The size of the training dataset.
            path_weights (str): Path to the model weights used for traning the model.
            save_path (str): Path to save the results.
            amino_acid (str): The amino acid to be modified.
            init_path (str, optional): Path for initialize/evaluate the model, to be able to load weights. Defaults to None.
            case (str, optional): Case type for handling different scenarios. Normal: Normal Clever Copy withput chaining.
                                Chaining_start: Start modification of chaining. Chaining: Modification in chain, not at start.
                                Unseen: Predict unseen modification.
            mods_to_add_to_alphabet (list, optional): List of modifications to add to the alphabet. Only needed for chaining. Defaults to None.
            ft_all_AA (bool, optional): Whether to fine-tune all amino acids. Defaults to None.
            save_output_to_file (bool, optional): Flag to save output to a file. Defaults to False.
    """
    def __init__(self, modification,train_path,test_path,val_path, trainsize, path_weights, save_path, amino_acid,init_path=None, case=None, mods_to_add_to_alphabet=None, ft_all_AA=None, save_output_to_file=False):
        self.modification = modification
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        self.save_path = save_path
        self.init_path = init_path
        self.trainsize = trainsize
        self.path_weights = path_weights
        self.amino_acid = amino_acid
        self.case = case
        self.mods_to_add_to_alphabet = mods_to_add_to_alphabet
        self.ft_all_AA = ft_all_AA
        self.save_output_to_file = save_output_to_file

        if self.init_path==None:
            self.init_path = val_path

        self.modified_AA = modification[-1]
        #self.shortcut_mod = utility.get_shortcut_of_mod(self.modification)

        self.config = {
            "batch_size": 512,
            "lr": 0.0001,
            "epochs": 3
        }

        self.setup_paths()
        self.create_alphabet()
        self.create_datasets()
        self.create_model()

    def setup_paths(self):
        """
        Setup the directory paths for saving results, weights, and validation/test outputs.
        """
        print("setup paths")

        if self.case == "unseen":
            self.save_final_results = self.save_path
            self.save_path_after = f'{self.save_path}/unseen/{self.modification}/after'
        else:
            self.save_final_results = self.save_path
            self.save_path_after = f'{self.save_path}/{self.modification}/after'

        self.save_path_test = f'{self.save_path_after}/test'
        self.save_path_val = f'{self.save_path_after}/validation'
        self.save_path_weights = f'{self.save_path_after}/weights'

        # Create directories
        pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.save_path_weights).mkdir(parents=True, exist_ok=True)

    def create_alphabet(self):
        """
        Add new modification(s) to the alphabet.
        """
        print("create alphabet")

        self.alphabet_AA = copy.deepcopy(dlomix.dlomix.constants.ALPHABET_UNMOD)
        
        if self.case == "unseen":
            for mod in self.mods_to_add_to_alphabet[:-1]:
                unimod_new = utility.get_modification_in_UNIMOD(modification=mod)
                self.alphabet_AA[unimod_new] = len(self.alphabet_AA) + 1
                #self.alphabet_AA[mod] = len(self.alphabet_AA) + 1
        elif self.case=='chaining':
            for mod in self.mods_to_add_to_alphabet:
                unimod_new = utility.get_modification_in_UNIMOD(modification=mod)
                self.alphabet_AA[unimod_new] = len(self.alphabet_AA) + 1
                #self.alphabet_AA[mod] = len(self.alphabet_AA) + 1
            unimod_new = utility.get_modification_in_UNIMOD(modification=self.modification)
            self.alphabet_AA[unimod_new] = len(self.alphabet_AA) + 1
        elif self.case=="normal" or self.case=="chaining_start" or self.case==None:
            unimod_new = utility.get_modification_in_UNIMOD(modification=self.modification)
            self.alphabet_AA[unimod_new] = len(self.alphabet_AA) + 1
            #self.alphabet_AA[self.modification] = len(self.alphabet_AA) + 1

        self.alphabet_keys = list(self.alphabet_AA.keys())
        print(self.alphabet_AA)

    def create_datasets(self):
        """
        Create datasets for training, validation, and testing using FragmentIonIntensityDataset.
        """
        print("create datasets")

        self.train = FragmentIonIntensityDataset(
            data_format="parquet",
            sequence_column="modified_sequence",
            label_column="intensities_raw",
            model_features=["precursor_charge_onehot", "collision_energy_aligned_normed"],
            max_seq_len=30,
            batch_size=self.config["batch_size"],
            alphabet=self.alphabet_AA,
            test_data_source=self.train_path,
            encoding_scheme="naive-mods",
            with_termini=False
        )

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

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config["lr"])
        self.model.compile(optimizer=optimizer, loss=masked_spectral_distance, metrics=["mse"])

    def train_model(self):
        """
        Train the model using the training dataset and save the best weights.
        """
        print("train model")

        weights_file = "./prosit_intensity_test.weights.h5"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            weights_file, save_best_only=True, save_weights_only=True
        )
        decay = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=10, verbose=1, min_lr=0
        )
        early_stop = tf.keras.callbacks.EarlyStopping(patience=20)
        callbacks = [checkpoint, early_stop, decay]

        self.model.evaluate(self.init_model.tensor_test_data,verbose=0)
        self.model.load_weights(self.path_weights)

        # Freeze all layers except embedding layer
        self.model.layers[0].trainable = True
        for layer in self.model.layers[1:]:
            layer.trainable = False

        self.model.fit(
            self.train.tensor_test_data,
            validation_data=self.validation.tensor_test_data,
            epochs=self.config["epochs"],
            callbacks=callbacks
        )

        self.model.save_weights(f"{self.save_path_weights}/{self.modification}_{self.amino_acid}_{self.trainsize}_after.weights.h5")

    def predict_and_save_results(self):
        """
        Predict the ion fragment intensities and calculate the spectral angle .
        """
        print("prediction after fine-tuning")

        predictions_val = self.model.predict(self.validation.tensor_test_data,verbose=0)
        df_spectral_angle_val = utility.get_spectral_angle(self.validation, predictions_val)
        self.spectral_angle_mean_val = df_spectral_angle_val['mean']

        predictions_test = self.model.predict(self.test.tensor_test_data,verbose=0)
        df_spectral_angle_test = utility.get_spectral_angle(self.test, predictions_test)
        self.spectral_angle_mean_test = df_spectral_angle_test['mean']

        if self.save_output_to_file:
            pathlib.Path(self.save_path_test).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.save_path_val).mkdir(parents=True, exist_ok=True)
            utility.write_spectral_angle_to_file(self.save_path_val, self.modification, self.trainsize, df_spectral_angle_val, self.amino_acid, "after")
            utility.write_spectral_angle_to_file(self.save_path_test, self.modification, self.trainsize, df_spectral_angle_test, self.amino_acid, "after")

        selected_row = df_spectral_angle_test.iloc[0]
        row_as_list = selected_row.astype(str).tolist()
        result_string = ','.join(row_as_list)

        if self.ft_all_AA == "false" and self.case != "unseen":
            with open(f"{self.save_final_results}/after_ft.csv", "a") as outfile:
                final_string = f"{self.amino_acid},{self.spectral_angle_mean_test},{self.modification},{self.trainsize}"
                outfile.write(final_string+"\n") 
        
        if self.case=="unseen":
            mods_so_far = self.mods_to_add_to_alphabet_list[:-1]
            '_'.join(mods_so_far)
            utility.write_spectral_angle_to_file2(f"{self.save_final_results}/unseen.csv",self.modification,self.trainsize,df_spectral_angle_test,self.amino_acid,f"{mods_so_far}")
        else:
            utility.write_spectral_angle_to_file2(f"{self.save_final_results}/finale_results.csv",self.modification,self.trainsize,df_spectral_angle_test,self.amino_acid,"after_chaining")

    def get_SA_after_ft(self):
        return self.spectral_angle_mean_test
    
    def get_weights_path_after_ft(self):
        return f"{self.save_path_weights}/{self.modification}_{self.amino_acid}_{self.trainsize}_after.weights.h5"
