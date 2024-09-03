
# Clever Copy Predictor

The Clever Copy predictor enables the prediction of ion fragment intensities of modified peptides by a novel approach of embedding layer initialization.

It combines transfer learning by using pre-trained weights on the prosite model for unmodified peptides with clever copying of embedding layer rows. 

It relies on the architecture and functions of the DLOmix tool (https://github.com/wilhelm-lab/dlomix.git).



## Documentation

It is necessary to be able to access the resources and functions of DLOmix.

_CleverCopyPredictorBeforeFineTuning_:
Finds the amino acid with the highest spectral angle before fine-tuning using the Clever Copy Apporach. 

_CleverCopyPredictorFineTuning_:
Trains a model consisting of prosite weights and a newly initialized line for the modified amino acid.

_Chaining_PTMs_:
Chaining of several models with different PTMs to better predict unknown PTMs.

In addition, there are two further classes for preprocessing the input data and creating combined training data sets for linking models.

_data_preprocessing_: Creates train, test and validation sets. The desired size of the train set can be passed as an optional parameter.

_trainsets_chaining_: Creates combined train sets consisting of several PTMs.

The weights of the Prosit model were reshaped (_build_model.py_) and saved as _base_prosit.weights.h5_.

Shortcuts for modifications are saved in _utils.py_ (_get_modification_in_UNIMOD_).
## Environment Variables

To run this project, you will need to use the environment file in folder _environment_.



## Usage/Examples

A notebook _example_notebook.ipyb_ with examples of using the CleverCopyPredictor and chaining models guiding you through the methods.

It is recommended to use the following dir structure:

-head_dir    
--input_files (processed, not the raw files)  
--chaining  
