import tensorflow as tf


def dropuout_layer(input, dropout_keep_prob, name):
    with tf.variable_scope("{0}_scope".format(name)):
        output = tf.nn.dropout(input, dropout_keep_prob, name='{0}_drop'.format(name))
        return output

def bidirectional_LSTM(input, lstm_hidden_layer_size, initializer, sequence_length=None, output_sequence=True, dropout_keep_prob = 1.0, name = "biLSTM"):
    with tf.variable_scope("{0}_scope".format(name)):
        if sequence_length == None:
            batch_size = 1
            sequence_length = tf.shape(input)[1]
            sequence_length = tf.expand_dims(sequence_length, axis=0, name='sequence_length')
        else:
            batch_size = tf.shape(sequence_length)[0]

        lstm_cell = {}
        initial_state = {}
        for direction in ["forward", "backward"]:
            with tf.variable_scope(direction):
                # LSTM cell
                lstm_cell[direction] = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(lstm_hidden_layer_size,
                                                                                     forget_bias=1.0,
                                                                                     initializer=initializer,
                                                                                     state_is_tuple=True)
                # initial state: http://stackoverflow.com/questions/38441589/tensorflow-rnn-initial-state
                initial_cell_state = tf.get_variable("initial_cell_state", shape=[1, lstm_hidden_layer_size],
                                                     dtype=tf.float32, initializer=initializer)
                initial_output_state = tf.get_variable("initial_output_state", shape=[1, lstm_hidden_layer_size],
                                                       dtype=tf.float32, initializer=initializer)
                c_states = tf.tile(initial_cell_state, tf.stack([batch_size, 1]))
                h_states = tf.tile(initial_output_state, tf.stack([batch_size, 1]))
                initial_state[direction] = tf.contrib.rnn.LSTMStateTuple(c_states, h_states)

        # sequence_length must be provided for tf.nn.bidirectional_dynamic_rnn due to internal bug
        outputs, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_cell["forward"],
                                                                lstm_cell["backward"],
                                                                input,
                                                                dtype=tf.float32,
                                                                sequence_length=sequence_length,
                                                                initial_state_fw=initial_state["forward"],
                                                                initial_state_bw=initial_state["backward"])



        if output_sequence == True:
            outputs_forward, outputs_backward = outputs
            bilstm_output = tf.concat([outputs_forward, outputs_backward], axis=2, name='output_sequence')
        else:
            final_states_forward, final_states_backward = final_states
            bilstm_output = tf.concat([final_states_forward[1], final_states_backward[1]], axis=1, name='output')

        bilstm_output_drop = dropuout_layer(bilstm_output, dropout_keep_prob=dropout_keep_prob, name="output")
    return bilstm_output_drop

def embedding_layer(input, vocab_size, embedding_size, initializer, name, trainable=True):
    with tf.variable_scope("{0}_scope".format(name)):
        if initializer is None:
            initializer = tf.contrib.layers.xavier_initializer()
        weights = tf.get_variable(
            "{0}_weights".format(name),
            shape=[vocab_size, embedding_size],
            initializer=initializer,
            trainable=trainable)
        output = tf.nn.embedding_lookup(weights, input)
        return output, weights



class MATEPC(object):
    """
    An LSTM architecture for named entity recognition.
    Uses a character embedding layer followed by an LSTM to generate vector representation from characters for each token.
    Then the character vector is concatenated with token embedding vector, which is input to another LSTM  followed by a CRF layer.
    """

    def get_inputs(self, config):
        # Placeholders for input, output and dropout
        self.input_word_indices = tf.placeholder(tf.int32, [None, config.max_length], name="input_word_indices")
        self.input_mask = tf.placeholder(tf.int32, [None, config.max_length], name="input_mask")
        self.input_sequence_length = tf.placeholder(tf.int32, [None], name="input_sequence_length")

        self.output_label_indices = tf.placeholder(tf.int32, [None, config.max_length], name="output_label_indices")
        self.dropout_keep_prob = tf.placeholder(tf.float32,
                                                name="dropout_keep_prob")  # placeholder for training: 0.5, testing: 1.


    def __init__(self, config):

        self.config = config
        self.verbose = False

        self.get_inputs(config=self.config)

        # Internal parameters
        initializer = tf.contrib.layers.xavier_initializer()

        # Word embedding layer
        self.embedded_words, self.word_embedding_weights = embedding_layer(input=self.input_word_indices,
                                              vocab_size=config.word_vocab_size,
                                              embedding_size=config.word_embedding_size,
                                              initializer=initializer, name="word_embedding", trainable=True) # [None, max_length, word_embedding_size]
        self.embedded_words_mask = tf.multiply(self.embedded_words, tf.expand_dims(tf.cast(self.input_mask, tf.float32), -1))
        self.embedded_words_drop = dropuout_layer(self.embedded_words_mask, dropout_keep_prob=self.dropout_keep_prob, name="embedded_words_mask")

        print("TFModel: LSTM")
        self.bilstm_output_drop = bidirectional_LSTM(input=self.embedded_words_drop,
                                                     lstm_hidden_layer_size=config.lstm_hidden_layer_size,
                                                     initializer=initializer,
                                                     output_sequence=True,
                                                     sequence_length=self.input_sequence_length,
                                                     dropout_keep_prob=self.dropout_keep_prob,
                                                     name="biLSTM")  # [None, max_length, 2*lstm_hidden_layer_size]

        # Needed only if Bidirectional LSTM is used for token level
        with tf.variable_scope("feedforward_after_lstm"):
            W = tf.get_variable(
                "W",
                shape=[config.lstm_hidden_layer_size*2, config.number_of_classes],
                initializer=initializer)
            b = tf.Variable(tf.constant(0.0, shape=[config.number_of_classes]), name="b")
            ff_outputs_1 = tf.tensordot(self.bilstm_output_drop, W, axes=1, name="output_before_softmax") + b  # (B, T, att_size)
            ff_outputs_1_reshape = tf.reshape(ff_outputs_1, [-1, config.max_length, config.number_of_classes])  # [None, max_length, 2*number_of_classes]
            ff_outputs_1_reshape_mask = tf.multiply(ff_outputs_1_reshape, tf.expand_dims(tf.cast(self.input_mask, tf.float32), -1))
            self.unary_scores = ff_outputs_1_reshape_mask
            self.predictions = tf.argmax(self.unary_scores, -1, name="predictions")

        with tf.variable_scope("crf"):
            # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/crf
            # Compute the log-likelihood of the gold sequences and keep the transition params for inference at test time.
            self.crf_transition_parameters = tf.get_variable(
                "transitions",
                shape=[config.number_of_classes, config.number_of_classes],
                initializer=initializer)
            self.log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(self.unary_scores, self.output_label_indices,
                                                                       self.input_sequence_length,
                                                                       transition_params=self.crf_transition_parameters)
            self.loss = tf.reduce_mean(-self.log_likelihood, name='cross_entropy_mean_loss')
            self.accuracy = tf.constant(1)

        self.reg = tf.constant(0, dtype=tf.float32)
        for variable in tf.trainable_variables():
            variables_name = variable.name
            if variables_name.find("W:") != -1 or variables_name.find("U:") != -1 or variables_name.find("W_0:") != -1:
                print(variables_name, variable.shape)
                self.reg += tf.nn.l2_loss(variable)

        self.train_op = tf.train.AdamOptimizer().minimize(self.loss + config.l2_lambda * self.reg)
        self.saver = tf.train.Saver()  # defaults to saving all variables

    def load_word_embedding(self, sess, initial_weights):
        sess.run(self.word_embedding_weights.assign(initial_weights))
