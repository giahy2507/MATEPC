
class ModelConfig(object):
    def __init__(self):

        # model config

        # max_sequence_length
        self.max_length = None

        # number of output classes
        self.number_of_classes = None

        self.embedding_name = "w2v"

        self.word_vocab_size = None
        self.word_embedding_size = None


        self.dropout_rate_train = 0.5
        self.dropout_rate_test = 1.0

        self.lstm_hidden_layer_size = None


        # training config

        # Batch size
        self.batch_size = 16

        # Optimizer for training the model.
        self.optimizer = 'adam'

        # Learning rate for the initial phase of training.
        self.learning_rate = 0.001

        # lambda for regularization
        self.l2_lambda = 0.001

        # The number of max epoch size
        self.max_epoch = 200

        # Parameters for early stopping
        self.early_stopping = True
        self.patience = 20

        # Fine-tune word embeddings
        self.train_embeddings = True

        # gradiend_clipvalue
        self.gradient_clipping_value = 5.0

    def parse_params(self, params_str="w2v,150,200,20,0.0010,20,0.001"):
        result = {}
        tokens = params_str.split(",")
        result["embedding_name"] = tokens[0]
        result["word_embedding_size"] = int(tokens[1])
        result["lstm_hidden_layer_size"] = int(tokens[2])
        result["batch_size"] = int(tokens[3])
        result["learning_rate"] = float(tokens[4])
        result["patience"] = int(tokens[5])
        result["l2_lambda"] = float(tokens[6])
        return result

    def adjust_params_follow_paramstr(self, paramstr):
        params_value = self.parse_params(paramstr)
        self.embedding_name = params_value["embedding_name"]
        self.word_embedding_size = int(params_value["word_embedding_size"])
        self.lstm_hidden_layer_size = int(params_value["lstm_hidden_layer_size"])
        self.batch_size = int(params_value["batch_size"])
        self.learning_rate = float(params_value["learning_rate"])
        self.patience = int(params_value["patience"])
        self.l2_lambda = float(params_value["l2_lambda"])

    def adjust_params_follow_preprocessor(self, preprocess):
        self.max_length = preprocess.max_length
        self.number_of_classes = preprocess.number_of_classes
        self.word_vocab_size = preprocess.word_vocab_size



