#imports
import numpy as np

"""
Link to code source https://stackoverflow.com/questions/58807467/tensorflow-keras-cudnngru-to-gru-conversion
Only needed for bias weights to reshape the array form. Used for
    - bidirectional forward gru bias: (1536,) to (2,768)
    - bidirectional backforward gru bias: (1536,) to (2,768)
    - gru deccoder bias bias: (3072,) to (2,1536)
All other weights of the weights file are in the correct shape.
"""

class Convert_rnn_weights:

    def __init__(self) -> None:
        pass


    #Converts weights for RNN layers between native and CuDNN format
    def _convert_rnn_weights(self, layer, weights):
        """Converts weights for RNN layers between native and CuDNN format.

        Input kernels for each gate are transposed and converted between Fortran
        and C layout, recurrent kernels are transposed. For LSTM biases are summed/
        split in half, for GRU biases are reshaped.

        Weights can be converted in both directions between `LSTM` and`CuDNNSLTM`
        and between `CuDNNGRU` and `GRU(reset_after=True)`. Default `GRU` is not
        compatible with `CuDNNGRU`.

        For missing biases in `LSTM`/`GRU` (`use_bias=False`) no conversion is made.

        Arguments:
            layer: Target layer instance.
            weights: List of source weights values (input kernels, recurrent
                kernels, [biases]) (Numpy arrays).

        Returns:
            A list of converted weights values (Numpy arrays).

        Raises:
            ValueError: for incompatible GRU layer/weights or incompatible biases
        """
    
        def transform_kernels(kernels, func, n_gates):
            """Transforms kernel for each gate separately using given function.

            Arguments:
                kernels: Stacked array of kernels for individual gates.
                func: Function applied to kernel of each gate.
                n_gates: Number of gates (4 for LSTM, 3 for GRU).

            Returns:
                Stacked array of transformed kernels.
            """
            return np.hstack([func(k) for k in np.hsplit(kernels, n_gates)])


        def transpose_input(from_cudnn):
            """Makes a function that transforms input kernels from/to CuDNN format.

            It keeps the shape, but changes between the layout (Fortran/C). Eg.:

            ```
            Keras                 CuDNN
            [[0, 1, 2],  <--->  [[0, 2, 4],
            [3, 4, 5]]          [1, 3, 5]]
            ```

            It can be passed to `transform_kernels()`.

            Arguments:
                from_cudnn: `True` if source weights are in CuDNN format, `False`
                    if they're in plain Keras format.

            Returns:
                Function that converts input kernel to the other format.
            """

            order = 'F' if from_cudnn else 'C'


            def transform(kernel):
                return kernel.T.reshape(kernel.shape, order=order)


            return transform


        target_class = layer.__class__.__name__


        #convert the weights between CuDNNLSTM and LSTM
        if target_class in ['LSTM', 'CuDNNLSTM'] and len(weights) == 3:
            # determine if we're loading a CuDNNLSTM layer
            # from the number of bias weights:
            # CuDNNLSTM has (units * 8) weights; while LSTM has (units * 4)
            # if there's no bias weight in the file, skip this conversion
            units = weights[1].shape[0]
            bias_shape = weights[2].shape
            n_gates = 4


            if bias_shape == (2 * units * n_gates,):
                source = 'CuDNNLSTM'
            elif bias_shape == (units * n_gates,):
                source = 'LSTM'
            else:
                raise ValueError('Invalid bias shape: ' + str(bias_shape))


            def convert_lstm_weights(weights, from_cudnn=True):
                """Converts the weights between CuDNNLSTM and LSTM.

                Arguments:
                    weights: Original weights.
                    from_cudnn: Indicates whether original weights are from CuDNN layer.

                Returns:
                    Updated weights compatible with LSTM.
                """

                # Transpose (and reshape) input and recurrent kernels
                kernels = transform_kernels(weights[0], transpose_input(from_cudnn),
                                            n_gates)
                recurrent_kernels = transform_kernels(weights[1], lambda k: k.T, n_gates)
                if from_cudnn:
                    # merge input and recurrent biases into a single set
                    biases = np.sum(np.split(weights[2], 2, axis=0), axis=0)
                else:
                    # Split single set of biases evenly to two sets. The way of
                    # splitting doesn't matter as long as the two sets sum is kept.
                    biases = np.tile(0.5 * weights[2], 2)
                return [kernels, recurrent_kernels, biases]


                if source != target_class:
                    weights = convert_lstm_weights(weights, from_cudnn=source == 'CuDNNLSTM')


        # convert the weights between CuDNNGRU and GRU(reset_after=True)
        if target_class in ['GRU', 'CuDNNGRU'] and len(weights) == 3:
            # We can determine the source of the weights from the shape of the bias.
            # If there is no bias we skip the conversion since
            # CuDNNGRU always has biases.

            units = weights[1].shape[0]
            bias_shape = weights[2].shape
            n_gates = 3


            def convert_gru_weights(weights, from_cudnn=True):
                """Converts the weights between CuDNNGRU and GRU.

                Arguments:
                    weights: Original weights.
                    from_cudnn: Indicates whether original weights are from CuDNN layer.

                Returns:
                    Updated weights compatible with GRU.
                """

                kernels = transform_kernels(weights[0], transpose_input(from_cudnn),
                                            n_gates)
                recurrent_kernels = transform_kernels(weights[1], lambda k: k.T, n_gates)
                biases = np.array(weights[2]).reshape((2, -1) if from_cudnn else -1)
                return [kernels, recurrent_kernels, biases]

                
            if bias_shape == (2 * units * n_gates,):
                source = 'CuDNNGRU'
            elif bias_shape == (2, units * n_gates):
                source = 'GRU(reset_after=True)'
            elif bias_shape == (units * n_gates,):
                source = 'GRU(reset_after=False)'
            else:
                raise ValueError('Invalid bias shape: ' + str(bias_shape))


            if target_class == 'CuDNNGRU':
                target = 'CuDNNGRU'
            elif layer.reset_after:
                target = 'GRU(reset_after=True)'
            else:
                target = 'GRU(reset_after=False)'


            # only convert between different types
            if source != target:
                types = (source, target)
            if 'GRU(reset_after=False)' in types:
                raise ValueError('%s is not compatible with %s' % types)
            if source == 'CuDNNGRU':
                weights = convert_gru_weights(weights, from_cudnn=True)
            elif source == 'GRU(reset_after=True)':
                weights = convert_gru_weights(weights, from_cudnn=False)
            else:
                print("same type")


        return weights