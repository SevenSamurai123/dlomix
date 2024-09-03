#imports
import h5py
import numpy as np
import tensorflow as tf

#dlomix imports
import dlomix

#own scripts import
from PTM.scripts.convert_rnn_weights import Convert_rnn_weights

"""
    Load a HDF5 weights file of base prosit and convert the weights to match dlmoix PrositIntensityPredictor model structure. Neccessary after tensorflow update.

    Arguments:
        hpf5_file: weights file of prosit in hdf5 format
        save_path: path to save the converted weights
"""


class handle_weights():

    def __init__(self,hpf5_file, save_path):
        super(handle_weights, self).__init__()
        
        self.hpf5_file = hpf5_file
        self.dict_AA_row = {}
        self.save_path = save_path
        self.embedding_weights_with_mod_row = None

        self.read_weights_file()
        self.create_AA_row_dict()
        self.rnn_weights = Convert_rnn_weights()

    """
    Read the a weights file in HDF5 format and assign the weights to variables
    """
    def read_weights_file(self):
        f = h5py.File(self.hpf5_file, 'r')

        self.embedding_weights = f['layers/embedding/vars']['0'][:]

        self.meta_dense_bias_weights = f['meta_encoder/layers/dense/vars']['1'][:]
        self.meta_dense_kernel_weights = f['meta_encoder/layers/dense/vars']['0'][:]

        self.encoder1_bw_bias_weights = f['sequence_encoder/layers/bidirectional/backward_layer/cell/vars']['2'][:]
        self.encoder1_bw_kernel_weights = f['sequence_encoder/layers/bidirectional/backward_layer/cell/vars']['0'][:]
        self.encoder1_bw_rkernel_weights = f['sequence_encoder/layers/bidirectional/backward_layer/cell/vars']['1'][:]

        self.encoder1_fw_bias_weights = f['sequence_encoder/layers/bidirectional/forward_layer/cell/vars']['2'][:]
        self.encoder1_fw_kernel_weights = f['sequence_encoder/layers/bidirectional/forward_layer/cell/vars']['0'][:]
        self.encoder1_fw_rkernel_weights = f['sequence_encoder/layers/bidirectional/forward_layer/cell/vars']['1'][:]

        self.encoder2_bias_weights = f['sequence_encoder/layers/gru/cell/vars']['2'][:]
        self.encoder2_kernel_weights = f['sequence_encoder/layers/gru/cell/vars']['0'][:]
        self.encoder2_rkernel_weights = f['sequence_encoder/layers/gru/cell/vars']['1'][:]

        self.decoder_bias_weights = f['layers/sequential/layers/gru/cell/vars']['2'][:]
        self.decoder_kernel_weights = f['layers/sequential/layers/gru/cell/vars']['0'][:]
        self.decoder_rkernel_weights = f['layers/sequential/layers/gru/cell/vars']['1'][:]

        self.dense_1_bias_weights = f['layers/sequential/layers/decoder_attention_layer/dense/vars']['1'][:]
        self.dense_1_kernel_weights = f['layers/sequential/layers/decoder_attention_layer/dense/vars']['0'][:]

        self.encoder_att_W_weights = f['layers/attention_layer/vars']['0'][:]
        self.encoder_att_b_weights = f['layers/attention_layer/vars']['1'][:]

        self.timedense_bias_weights = f['regressor/layers/time_distributed/layer/vars']['1'][:]
        self.timedense_kernel_weights = f['regressor/layers/time_distributed/layer/vars']['0'][:]

        f.close()


    """
    Add x new rows to the embedding matrix and assign all weights to the corresponding layers.
    Than save the model weights as file.

    Arguments:
        model: A evaluated keras model able to load the weights.

    Returns:
        Weights file containing the weights with a x new embedding rows.
        File saved as 'model_with_weights.weights.h5'.
    """
    def save_custom_weights_to_file(self,model, numNewRows):

        #Instantiate random uniform distribution 
        rows=[]
        for i in range(0,numNewRows):
            initializer = tf.keras.initializers.RandomUniform(
                            minval=-0.05, maxval=0.05, seed=None
                            )
            new_row = initializer(shape=(1, 32))
            rows.append(new_row)

        rows_matrix=np.concatenate(rows, axis=0)

        #append new array at the end of the embedding matrix
        self.embedding_weights_with_mod_row = np.vstack((self.embedding_weights,rows_matrix))
        model.layers[0].set_weights([self.embedding_weights_with_mod_row])   
            
        #Sequence Encoder
        model.layers[1].set_weights([self.meta_dense_kernel_weights, self.meta_dense_bias_weights])  

        model.layers[2].set_weights([self.encoder1_fw_kernel_weights, 
                                            self.encoder1_fw_rkernel_weights, 
                                            self.encoder1_fw_bias_weights, 
                                            self.encoder1_bw_kernel_weights, 
                                            self.encoder1_bw_rkernel_weights, 
                                            self.encoder1_bw_bias_weights, 
                                            self.encoder2_kernel_weights, 
                                            self.encoder2_rkernel_weights, 
                                            self.encoder2_bias_weights])  

        model.layers[3].set_weights([self.decoder_kernel_weights, 
                                            self.decoder_rkernel_weights,
                                            self.decoder_bias_weights, 
                                            self.dense_1_kernel_weights, 
                                            self.dense_1_bias_weights])  


        #AttentionLayer
        model.layers[4].set_weights([self.encoder_att_W_weights, 
                                            self.encoder_att_b_weights])  

        #Regressor
        model.layers[6].set_weights([self.timedense_kernel_weights, 
                                            self.timedense_bias_weights])  
    
        
        #save model weights
        model.save_weights(self.save_path)


    """
    Create a dictionary of amino acids and their emnbedding row.
        Key: amino acid, string
        Value: embedding row, array of shape (1,32)
    """
    def create_AA_row_dict(self):
        weights_matrix  = self.embedding_weights
        aa = dlomix.dlomix.constants.ALPHABET_UNMOD.keys()
        self.dict_AA_row = { amino : row for amino,row in zip(aa,weights_matrix[1:])}
        #self.dict_AA_row[self.modification] = weights_matrix[-1]


    """
    Get the embedding row of a certain amino acid.

    Arguments:
        AA: amino acid as string, in one letter code. Corresponding to the dlomix.constants.ALPHABET.
    
    Returns:
        Array of shape (1,32) of the input amino acid
    """
    def get_embedding_weights_of_AA(self,AA):
        return self.dict_AA_row[AA]
    

    """
    Get the entire embedding layer back as array.

    Returns:
        Array of shape embedding layer
    """
    def get_embedding_weights(self):
        return self.embedding_weights_with_mod_row
    
    
    def get_embedding_weights_raw(self):
        return self.embedding_weights