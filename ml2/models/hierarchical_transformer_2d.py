"""Hierarchical Transformer implementation

The hierarchical Transformer architecture was introduced in https://arxiv.org/abs/2006.09265
"""

import tensorflow as tf

from ..layers import positional_encoding as pe
from . import transformer
from .beam_search import BeamSearch, flatten_beam_dim


def create_model(params, training, custom_pos_enc=False, attn_weights=False):
    """
    Args:
        params: dict, hyperparameter dictionary
        training: bool, whether model is called in training mode or not
        custom_pos_enc, bool, whether a custom positional encoding is provided as additional input
        attn_weights: bool, whether attention weights are part of the output
    """
    params["return_attn_weights"] = attn_weights
    input = tf.keras.layers.Input((None, None), dtype=tf.int32, name="input")
    transformer_inputs = {"input": input}
    model_inputs = [input]
    if custom_pos_enc:
        positional_encoding = tf.keras.layers.Input(
            (None, None, None), dtype=tf.float32, name="positional_encoding"
        )
        transformer_inputs["positional_encoding"] = positional_encoding
        model_inputs.append(positional_encoding)
    if training:
        target = tf.keras.layers.Input((None,), dtype=tf.int32, name="target")
        transformer_inputs["target"] = target
        model_inputs.append(target)
    hierarchical_transformer = HierarchicalTransformer(params)
    if training:
        # do not provide training argument so keras fit method can set it
        predictions, _ = hierarchical_transformer(transformer_inputs)
        predictions = transformer.TransformerMetricsLayer(params)([predictions, target])
        model = tf.keras.Model(model_inputs, predictions)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        mask = tf.cast(
            tf.math.logical_not(tf.math.equal(target, params["target_pad_id"])),
            params["dtype_float"],
        )
        loss = tf.keras.layers.Lambda(lambda x: loss_object(x[0], x[1], x[2]))(
            (target, predictions, mask)
        )
        model.add_loss(loss)
        return model
    else:
        # do not provide training argument so keras fit method can set it
        results = hierarchical_transformer(transformer_inputs)
        if attn_weights:
            outputs, scores, enc_attn_weights_local, enc_attn_weights_global, dec_attn_weights = (
                results["outputs"],
                results["scores"],
                results["enc_attn_weights_local"],
                results["enc_attn_weights_global"],
                results["dec_attn_weights"],
            )
            return tf.keras.Model(
                model_inputs,
                [
                    outputs,
                    scores,
                    enc_attn_weights_local,
                    enc_attn_weights_global,
                    dec_attn_weights,
                ],
            )
        else:
            outputs, scores = results["outputs"], results["scores"]
            return tf.keras.Model(model_inputs, [outputs, scores])


class HierarchicalTransformer(tf.keras.Model):
    def __init__(self, params):
        """
        Args:
            params: hyperparameter dictionary containing the following keys:

                d_embed_enc: int, dimension of encoder embedding
                d_ff_enc_d0: int, hidden dimension of dimension 0 encoder feed-forward networks
                d_ff_enc_d1: int, hidden dimension of dimension 1 encoder feed-forward networks
                dropout_enc: float, percentage of dropped out encoder units
                ff_activation_enc_d0: string, activation function used in dimension 0 encoder feed-forward networks
                ff_activation_enc_d1: string, activation function used in dimension 1 encoder feed-forward networks
                num_heads_enc_d0: int, number of dimension 0 encoder attention heads
                num_heads_enc_d1: int, number of dimension 1 encoder attention heads
                num_layers_enc_d0: int, number of dimension 0 encoder layer
                num_layers_enc_d0: int, number of dimension 1 encoder layer
                input_length: (int, int), dimensions of 2 dimensional input
                input_pad_id: int, encodes the padding token for inputs
                input_vocab_size: int, size of input vocabulary



                alpha: float, strength of normalization in beam search algorithm
                beam_size: int, number of beams kept by beam search algorithm
                d_embed_dec: int, dimension of decoder embedding
                d_ff_dec: int, hidden dimension of decoder feed-forward networks
                dropout_dec: float, percentage of dropped out decoder units
                dtype_float: tf.dtypes.Dtype(), datatype for floating point computations
                ff_activation_dec: string, activation function used in decoder feed-forward networks
                max_decode_length: int, maximum length of target sequence
                num_heads_dec: int, number of decoder attention heads
                num_layers_dec: int, number of decoder layer
                target_eos_id: int, encodes the end of string token for targets
                target_pad_id: int, encodes the padding token for targets
                target_start_id: int, encodes the start token for targets
                target_vocab_size: int, size of target vocabulary
        """
        super().__init__()
        self.params = params

        self.encoder_embedding = tf.keras.layers.Embedding(
            params["input_vocab_size"], params["d_embed_enc"]
        )
        # encoder_positional_encoding_d1 = pe.positional_encoding(
        #    params['input_length'][1], params['d_embed_enc'])
        # self.encoder_positional_encoding = tf.repeat(
        #    tf.expand_dims(encoder_positional_encoding_d1, axis=0),
        #    repeats=[params['input_length'][0]],
        #    axis=0)
        self.encoder_dropout = tf.keras.layers.Dropout(params["dropout_enc"])

        self.encoder_stack_d0 = transformer.TransformerEncoder(
            {
                "d_embed_enc": params["d_embed_enc"],
                "d_ff": params["d_ff_enc_d0"],
                "dropout": params["dropout_enc"],
                "ff_activation": params["ff_activation_enc_d0"],
                "num_heads": params["num_heads_enc_d0"],
                "num_layers_enc": params["num_layers_enc_d0"],
            }
        )
        self.encoder_stack_d1 = transformer.TransformerEncoder(
            {
                "d_embed_enc": params["d_embed_enc"],
                "d_ff": params["d_ff_enc_d1"],
                "dropout": params["dropout_enc"],
                "ff_activation": params["ff_activation_enc_d1"],
                "num_heads": params["num_heads_enc_d1"],
                "num_layers_enc": params["num_layers_enc_d1"],
            }
        )

        self.decoder_embedding = tf.keras.layers.Embedding(
            params["target_vocab_size"], params["d_embed_dec"]
        )
        self.decoder_positional_encoding = pe.positional_encoding(
            params["max_decode_length"], params["d_embed_dec"]
        )
        self.decoder_dropout = tf.keras.layers.Dropout(params["dropout_dec"])

        self.decoder_stack = transformer.TransformerDecoder(
            {
                "d_embed_dec": params["d_embed_dec"],
                "d_ff": params["d_ff_dec"],
                "dropout": params["dropout_dec"],
                "ff_activation": params["ff_activation_dec"],
                "num_heads": params["num_heads_dec"],
                "num_layers_dec": params["num_layers_dec"],
            }
        )

        self.final_projection = tf.keras.layers.Dense(params["target_vocab_size"])
        self.softmax = tf.keras.layers.Softmax()

    def get_config(self):
        return {"params": self.params}

    def call(self, inputs, training):
        """
        Args:
            inputs: dictionary that contains the following (optional) keys:
                input: int tensor with shape (batch_size, input_length[0], input_length[1])
                (positional_encoding: float tensor with shape (batch_size, input_length[0], input_length[1], d_embed_enc), custom postional encoding)
                (target: int tensor with shape (batch_size, target_length))
            training: bool, whether model is called in training mode or not
        """
        input = inputs["input"]

        input_padding_mask = tf.cast(
            tf.math.equal(input, self.params["input_pad_id"]), self.params["dtype_float"]
        )
        input_padding_mask = input_padding_mask[:, tf.newaxis, tf.newaxis, :, :]

        if "positional_encoding" in inputs:
            positional_encoding = inputs["positional_encoding"]
        else:
            raise NotImplementedError
            # seq_len = tf.shape(input)[1]
            # positional_encoding = self.encoder_positional_encoding[:, :
            #                                                       seq_len, :]

        encoder_output, enc_attn_weights_local, enc_attn_weights_global = self.encode(
            input, input_padding_mask, positional_encoding, training
        )

        input_shape = tf.shape(input)
        batch_size = input_shape[0]
        input_length_d0 = self.params["input_length"][0]
        input_length_d1 = self.params["input_length"][1]
        if self.params["fix_d1_embed"]:
            input_padding_mask = input_padding_mask[:, :, :, :, 0]
            input_length_d1 = 1
        input_padding_mask = tf.reshape(
            input_padding_mask, [batch_size, 1, 1, input_length_d0 * input_length_d1]
        )

        if "target" in inputs:
            target = inputs["target"]
            return self.decode(target, encoder_output, input_padding_mask, training)
        else:
            return self.predict(
                encoder_output,
                enc_attn_weights_local,
                enc_attn_weights_global,
                input_padding_mask,
                training,
            )

    def encode(self, inputs, padding_mask, positional_encoding, training):
        """
        Args:
            inputs: int tensor with shape (batch_size, input_length[0], input_length[1])
            padding_mask: float tensor with shape (batch_size, 1, 1, input_length[0], input_length[1])
            positional_encoding: float tensor with shape (batch_size, input_length[0], input_length[1], d_embed_enc)
            training: boolean, specifies whether in training mode or not
        """
        input_embedding = self.encoder_embedding(inputs)
        input_embedding *= tf.math.sqrt(
            tf.cast(self.params["d_embed_enc"], self.params["dtype_float"])
        )
        input_embedding += positional_encoding
        input_embedding = self.encoder_dropout(input_embedding, training=training)
        # reshape to (batch_size * input_length[0], input_length[1], d_embed_enc)
        input_shape = tf.shape(input_embedding)
        batch_size = input_shape[0]
        input_length_d0 = self.params["input_length"][0]
        input_length_d1 = self.params["input_length"][1]
        d_embed_enc = self.params["d_embed_enc"]
        input_embedding_d1 = tf.reshape(
            input_embedding, [batch_size * input_length_d0, input_length_d1, d_embed_enc]
        )
        padding_mask_d1 = tf.reshape(
            padding_mask, [batch_size * input_length_d0, 1, 1, input_length_d1]
        )

        encoder_output_d1, attn_weights_d1 = self.encoder_stack_d1(
            input_embedding_d1, padding_mask_d1, training
        )

        if self.params["fix_d1_embed"]:
            encoder_output_d1 = encoder_output_d1[:, 0, :]
            padding_mask = padding_mask[:, :, :, :, 0]
            input_length_d1 = 1

        # reshape to (batch_size, input_length[0] * input_length[1], d_embed_enc)
        input_embedding_d0 = tf.reshape(
            encoder_output_d1, [batch_size, input_length_d0 * input_length_d1, d_embed_enc]
        )
        padding_mask_d0 = tf.reshape(
            padding_mask, [batch_size, 1, 1, input_length_d0 * input_length_d1]
        )

        encoder_output_d0, attn_weights_d0 = self.encoder_stack_d0(
            input_embedding_d0, padding_mask_d0, training
        )

        return encoder_output_d0, attn_weights_d1, attn_weights_d0

    def decode(self, target, encoder_output, input_padding_mask, training):
        """
        Args:
            target: int tensor with shape (bath_size, target_length) including start id at first position
            encoder_output: float tensor with shape (batch_size, input_length, d_embedding)
            input_padding_mask: float tensor with shape (batch_size, 1, 1, input_length)
            training: boolean, specifies whether in training mode or not
        """
        target_length = tf.shape(target)[1]
        look_ahead_mask = transformer.create_look_ahead_mask(
            target_length, self.params["dtype_float"]
        )
        target_padding_mask = transformer.create_padding_mask(
            target, self.params["input_pad_id"], self.params["dtype_float"]
        )
        look_ahead_mask = tf.maximum(look_ahead_mask, target_padding_mask)

        # shift targets to the right, insert start_id at first postion, and remove last element
        target = tf.pad(target, [[0, 0], [1, 0]], constant_values=self.params["target_start_id"])[
            :, :-1
        ]
        target_embedding = self.decoder_embedding(
            target
        )  # (batch_size, target_length, d_embedding)
        target_embedding *= tf.math.sqrt(
            tf.cast(self.params["d_embed_dec"], self.params["dtype_float"])
        )

        target_embedding += self.decoder_positional_encoding[:, :target_length, :]
        decoder_embedding = self.decoder_dropout(target_embedding, training=training)
        decoder_output, attn_weights = self.decoder_stack(
            decoder_embedding, encoder_output, look_ahead_mask, input_padding_mask, training
        )
        output = self.final_projection(decoder_output)
        probs = self.softmax(output)
        return probs, attn_weights

    def predict(
        self,
        encoder_output,
        enc_attn_weights_local,
        enc_attn_weights_global,
        input_padding_mask,
        training,
    ):
        """
        Args:
            encoder_output: float tensor with shape (batch_size, input_length, d_embedding)
            encoder_attn_weights: dictionary, self attention weights of the encoder
            input_padding_mask: flaot tensor with shape (batch_size, 1, 1, input_length)
            training: boolean, specifies whether in training mode or not
        """
        batch_size = tf.shape(encoder_output)[0]

        def logits_fn(ids, i, cache):
            """
            Args:
                ids: int tensor with shape (batch_size * beam_size, index + 1)
                index: int, current index
                cache: dictionary storing encoder output, previous decoder attention values
            Returns:
                logits with shape (batch_size * beam_size, vocab_size) and updated cache
            """
            # set input to last generated id
            decoder_input = ids[:, -1:]
            decoder_input = self.decoder_embedding(decoder_input)
            decoder_input *= tf.math.sqrt(
                tf.cast(self.params["d_embed_dec"], self.params["dtype_float"])
            )
            decoder_input += self.decoder_positional_encoding[:, i : i + 1, :]
            # dropout only makes sense if needs to be tested in training mode
            # think about removing dropout
            decoder_input = self.decoder_dropout(decoder_input, training=training)
            look_ahead_mask = transformer.create_look_ahead_mask(
                self.params["max_decode_length"], self.params["dtype_float"]
            )
            self_attention_mask = look_ahead_mask[:, :, i : i + 1, : i + 1]
            decoder_output, attn_weights = self.decoder_stack(
                decoder_input,
                cache["encoder_output"],
                self_attention_mask,
                cache["input_padding_mask"],
                training,
                cache,
            )

            logits = self.final_projection(decoder_output)
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache

        initial_ids = tf.ones([batch_size], dtype=tf.int32) * self.params["target_start_id"]

        num_heads = self.params["num_heads_dec"]
        d_heads = self.params["d_embed_dec"] // num_heads
        # create cache structure for decoder attention
        cache = {
            "layer_%d"
            % layer: {
                "keys": tf.zeros(
                    [batch_size, 0, num_heads, d_heads], dtype=self.params["dtype_float"]
                ),
                "values": tf.zeros(
                    [batch_size, 0, num_heads, d_heads], dtype=self.params["dtype_float"]
                ),
            }
            for layer in range(self.params["num_layers_dec"])
        }
        # add encoder output to cache
        cache["encoder_output"] = encoder_output
        cache["input_padding_mask"] = input_padding_mask

        beam_search = BeamSearch(logits_fn, batch_size, self.params)
        decoded_ids, scores = beam_search.search(initial_ids, cache)

        if self.params["return_attn_weights"]:

            # computer decoder attention weights
            _, dec_attn_weights = self.decode(
                flatten_beam_dim(decoded_ids), encoder_output, input_padding_mask, training
            )

            return {
                "outputs": decoded_ids,
                "scores": scores,
                "enc_attn_weights_local": enc_attn_weights_local,
                "enc_attn_weights_global": enc_attn_weights_global,
                "dec_attn_weights": dec_attn_weights,
            }

        else:

            return {"outputs": decoded_ids, "scores": scores}
