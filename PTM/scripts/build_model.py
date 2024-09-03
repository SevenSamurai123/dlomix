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


class handle_base_prosit_weights():

    def __init__(self,hpf5_file, save_path):
        super(handle_base_prosit_weights, self).__init__()
        
        self.hpf5_file = hpf5_file
        self.dict_AA_row = {}
        self.save_path = save_path
        #self.modification = modification[1].lower()
        self.embedding_weights_with_mod_row = None

        self.read_weights_file()
        self.create_AA_row_dict()
        self.rnn_weights = Convert_rnn_weights()

    """
    Read the base prosit weights file in HDF5 format and assign the weights to variables
    """
    def read_weights_file(self):
        f = h5py.File(self.hpf5_file, 'r')

        group = f['model_weights']
        embedding = group['embedding']
        meta_dense = group['meta_dense']
        encoder1 = group['encoder1']
        encoder2 = group['encoder2']
        decoder = group['decoder']
        dense_1 = group['dense_1']
        encoder_att = group['encoder_att']
        timedense = group['timedense']

        self.embedding_weights = embedding['embedding']['embeddings:0'][:]

        self.meta_dense_bias_weights = meta_dense['meta_dense']['bias:0'][:]
        self.meta_dense_kernel_weights = meta_dense['meta_dense']['kernel:0'][:]

        self.encoder1_bw_bias_weights = encoder1['encoder1']['backward_encoder1_gru']['bias:0'][:]
        self.encoder1_bw_kernel_weights = encoder1['encoder1']['backward_encoder1_gru']['kernel:0'][:]
        self.encoder1_bw_rkernel_weights = encoder1['encoder1']['backward_encoder1_gru']['recurrent_kernel:0'][:]

        self.encoder1_fw_bias_weights = encoder1['encoder1']['forward_encoder1_gru']['bias:0'][:]
        self.encoder1_fw_kernel_weights = encoder1['encoder1']['forward_encoder1_gru']['kernel:0'][:]
        self.encoder1_fw_rkernel_weights = encoder1['encoder1']['forward_encoder1_gru']['recurrent_kernel:0'][:]

        self.encoder2_bias_weights = encoder2['encoder2']['bias:0'][:]
        self.encoder2_kernel_weights = encoder2['encoder2']['kernel:0'][:]
        self.encoder2_rkernel_weights = encoder2['encoder2']['recurrent_kernel:0'][:]

        self.decoder_bias_weights = decoder['decoder']['bias:0'][:]
        self.decoder_kernel_weights = decoder['decoder']['kernel:0'][:]
        self.decoder_rkernel_weights = decoder['decoder']['recurrent_kernel:0'][:]

        self.dense_1_bias_weights = dense_1['dense_1']['bias:0'][:]
        self.dense_1_kernel_weights = dense_1['dense_1']['kernel:0'][:]

        self.encoder_att_W_weights = encoder_att['encoder_att']['encoder_att_W:0'][:]
        self.encoder_att_b_weights = encoder_att['encoder_att']['encoder_att_b:0'][:]

        self.timedense_bias_weights = timedense['timedense']['bias:0'][:]
        self.timedense_kernel_weights = timedense['timedense']['kernel:0'][:]

        f.close()


    """
    Convert the weights into the correct format and assign them to the corresponding model layers.
    Than save the model weights as file.

    Arguments:
        model: A trained keras model on 1 epoch to be able to load the weights.

    Returns:
        Weights file containing the base prosit weights in the correct shape.
        File saved as 'model_with_weights.weights.h5'.
    """
    def save_custom_weights_to_file(self,model):
        model.layers[0].set_weights([self.embedding_weights])   
            
        #Sequence Encoder
        model.layers[1].set_weights([self.meta_dense_kernel_weights, self.meta_dense_bias_weights])  

        #Encoder
        kernel_fw, recurrent_kernel_fw, bias_fw = self.rnn_weights._convert_rnn_weights(
            layer=model.layers[2].layers[2],
            weights=[
                self.encoder1_fw_kernel_weights,
                self.encoder1_fw_rkernel_weights,
                self.encoder1_fw_bias_weights,
            ],
        )
        
        
        kernel_bw, recurrent_kernel_bw, bias_bw = self.rnn_weights._convert_rnn_weights(
            layer=model.layers[2].layers[2],
            weights=[
                self.encoder1_bw_kernel_weights,
                self.encoder1_bw_rkernel_weights,
                self.encoder1_bw_bias_weights,
            ],
        )


        kernel_enc2, recurrent_kernel_enc2, bias_enc2 = self.rnn_weights._convert_rnn_weights(
            layer=model.layers[2].layers[2],
            weights=[
                self.encoder2_kernel_weights,
                self.encoder2_rkernel_weights,
                self.encoder2_bias_weights,
            ],
        )

        model.layers[2].set_weights([kernel_fw, 
                                            recurrent_kernel_fw, 
                                            bias_fw, 
                                            kernel_bw, 
                                            recurrent_kernel_bw, 
                                            bias_bw, 
                                            kernel_enc2, 
                                            recurrent_kernel_enc2, 
                                            bias_enc2])  


        # Decoder
        kernel_dec, recurrent_kernel_dec, bias_dec = self.rnn_weights._convert_rnn_weights(
            layer=model.layers[3].layers[0],
            weights=[
                self.decoder_kernel_weights,
                self.decoder_rkernel_weights,
                self.decoder_bias_weights,
            ],
        )

        model.layers[3].set_weights([kernel_dec, 
                                            recurrent_kernel_dec,
                                            bias_dec, 
                                            self.dense_1_kernel_weights, 
                                            self.dense_1_bias_weights])  


        #AttentionLayer
        model.layers[4].set_weights([self.encoder_att_W_weights, 
                                            self.encoder_att_b_weights])  

        #FusionLayer
        #has no weights

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