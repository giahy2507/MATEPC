import argparse
import os
import numpy as np
# np.random.seed(0)
import tensorflow as tf
# tf.set_random_seed(0)
from sklearn.model_selection import KFold

import pickle
import copy

from  myutils import collect_data_infor_from_tsv, load_word_embeddings

from config import ModelConfig
from features2 import WordPOSPreprocessor

from evaluation import ATEPCEvaluator, ATEPCNewEvaluator


from context_attention2 import AspectLSTM

import time


def create_data_object(X_train, Y_train, X_valid, Y_valid, X_test, Y_test):
    results = {}
    results['X_train'] = X_train
    results['Y_train'] = Y_train
    results['X_valid'] = X_valid
    results['Y_valid'] = Y_valid
    results['X_test'] = X_test
    results['Y_test'] = Y_test
    print("Data train: ", X_train[0].shape, Y_train.shape)
    print("Data valid: ", X_valid[0].shape, Y_valid.shape)
    print("Data  test: ", X_test[0].shape, Y_test.shape)
    return results


def train_step(sess, model, model_config, data, data_type):
    X_train = data["X_{0}".format(data_type)]
    Y_train = data["Y_{0}".format(data_type)]
    total_loss = []
    no_batch = int(Y_train.shape[0]/model_config.batch_size)
    minibatch_fold = KFold(n_splits=no_batch, shuffle=True)
    for train_index, valid_index in minibatch_fold.split(Y_train):
        feed_dict = {
            model.input_word_indices:       X_train[0][valid_index],
            model.input_pos_indices:        X_train[1][valid_index],
            model.input_mask:               X_train[3][valid_index],
            model.input_sequence_length:    X_train[2][valid_index],
            model.input_punc_mask:          X_train[4][valid_index],
            model.output_label_indices:     Y_train[valid_index],
            model.dropout_keep_prob:        0.5
        }
        _, loss, crf_transition_parameters = sess.run([model.train_op, model.loss, model.crf_transition_parameters], feed_dict)
        total_loss.append(loss)
        # print(loss)
    return crf_transition_parameters, sum(total_loss)/len(total_loss)

def get_entities(seq):
    """Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> print(get_entities(seq))
        [('PER', 0, 2), ('LOC', 3, 4)]
    """
    i = 0
    chunks = []
    seq = seq + ['O']  # add sentinel
    types = [tag.split('-')[-1] for tag in seq]
    while i < len(seq):
        if seq[i].startswith('B'):
            for j in range(i+1, len(seq)):
                if seq[j].startswith('I') and types[j] == types[i]:
                    continue
                break
            chunks.append((types[i], i, j))
            i = j
        else:
            i += 1
    return chunks

def f1_score(y_true, y_pred):
    """Evaluates f1 score.

    Args:
        y_true (list): true labels.
        y_pred (list): predicted labels.
        sequence_lengths (list): sequence lengths.

    Returns:
        float: f1 score.

    Example:
        >>> y_true = []
        >>> y_pred = []
        >>> sequence_lengths = []
        >>> print(f1_score(y_true, y_pred, sequence_lengths))
        0.8
    """
    correct_preds, total_correct, total_preds = 0., 0., 0.
    for lab, lab_pred in zip(y_true, y_pred):
        lab_chunks = set(get_entities(lab))
        lab_pred_chunks = set(get_entities(lab_pred))

        correct_preds += len(lab_chunks & lab_pred_chunks)
        total_preds += len(lab_pred_chunks)
        total_correct += len(lab_chunks)

    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    return f1

def predict_step(sess, model, p, data, data_type, crf_transition_parameters):
    X = data["X_{0}".format(data_type)]
    Y = data["Y_{0}".format(data_type)]
    ys_pred = []
    ys_true = []
    losses = []
    for i in range(Y.shape[0]):
        feed_dict = {
            model.input_word_indices:  X[0][i:i+1,:],
            model.input_pos_indices: X[1][i:i+1],
            model.input_mask: X[3][i:i+1],
            model.input_sequence_length: X[2][i:i+1],
            model.input_punc_mask: X[4][i:i+1],
            model.output_label_indices: Y[i:i+1],
            model.dropout_keep_prob: 1.0
        }
        unary_scores, loss = sess.run([model.unary_scores, model.loss], feed_dict)
        losses.append(loss)
        # print(unary_scores.shape)
        # print(unary_scores[0].shape)
        # print(X[0][i:i+1,:])
        # print(X[2][i])
        unary_scores_i = unary_scores[0][:X[2][i],:]
        # print(unary_scores_i.shape)

        y_pred, _ = tf.contrib.crf.viterbi_decode(unary_scores_i, crf_transition_parameters)
        y_true = list(Y[i][:X[2][i]])

        y_true_inversed =  p.inverse_transform(y_true)
        y_pred_inversed = p.inverse_transform(y_pred)
        ys_pred.append(y_pred_inversed)
        ys_true.append(y_true_inversed)
        assert len(y_pred) == len(y_true)
    f1 = f1_score(ys_pred, ys_true)
    losses = np.array(losses)
    losses_avg = np.mean(losses)
    return f1, ys_pred, ys_true, losses_avg


def save_result(result, i = -1):
    results = load_result()
    if results is None:
        results = []
    if i == -1:
        results.append(result)
    else:
        results[i] = result
    with open("logs/resultfinal.pickle", "wb") as f:
        pickle.dump(results, f)

def load_result():
    if os.path.isfile("logs/resultfinal.pickle") is False:
        return None
    with open("logs/resultfinal.pickle", "rb") as f:
        return pickle.load(f)


def write_result(fo_path, sents, ys_true, ys_pred):
    with open(fo_path, mode="w") as f:
        for sent, y_true, y_pred in zip(sents, ys_true, ys_pred):
            assert len(sent) == len(y_true) == len(y_pred)
            for word, y_t, y_p in zip(sent, y_true, y_pred):
                f.write("{0}\t{1}\t{2}\n".format(word, y_t, y_p))
            f.write("\n")

def train_model(keras_model_name= "WP", data_name="laptops", task_name="ATEPC2", hand_features = None, params_str = "w2v,100,100,100,20,0.0010", attention_status = "Attention_Before_LSTM"):
    DATA_ROOT = os.getcwd() + '/data'
    SAVE_ROOT = os.getcwd() + '/models'  # trained models
    LOG_ROOT = os.getcwd() + '/logs'  # checkpoint, tensorboard

    print("-----{0}-----{1}-----{2}-----{3}-----{4}".format(task_name, data_name, keras_model_name, hand_features, params_str))
    # ----- create save directory -----
    save_path = SAVE_ROOT + "/{0}/{1}".format(data_name, task_name)
    if not os.path.exists(SAVE_ROOT + "/{0}".format(data_name)):
        os.makedirs(SAVE_ROOT + "/{0}".format(data_name))
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    # ----- load raw data -----
    train_path = os.path.join(DATA_ROOT, '{0}.{1}.train.tsv'.format(data_name, task_name))
    test_path = os.path.join(DATA_ROOT, '{0}.{1}.test.tsv'.format(data_name, task_name))
    # train set
    if task_name == "ATE":
        sents1, poses1, dep_idxs1, dep_relations1, labels1, preds1 = collect_data_infor_from_tsv(train_path, keep_conflict=True)
    else:
        sents1, poses1, dep_idxs1, dep_relations1, labels1, preds1 = collect_data_infor_from_tsv(train_path, keep_conflict=False)
    X1_train_valid = sents1
    X2_train_valid = np.asarray(list(zip(poses1, dep_idxs1, dep_relations1)))
    Y_train_valid = labels1
    # test set
    sents2, poses2, dep_idxs2, dep_relations2, labels2, preds2 = collect_data_infor_from_tsv(test_path, keep_conflict=True)
    X1_test = sents2
    X2_test = np.asarray(list(zip(poses2, dep_idxs2, dep_relations2)))
    Y_test_origin = labels2
    # train + test for counting vocab size
    X1_train_test = np.concatenate((X1_train_valid, X1_test), axis=0)
    X2_train_test = np.concatenate((X2_train_valid, X2_test), axis=0)
    Y_train_test = np.concatenate((Y_train_valid, Y_test_origin), axis=0)

    # ----- Model Config
    model_config = ModelConfig()
    model_config.attention_status = attention_status
    model_config.learning_feature = keras_model_name
    model_config.adjust_params_follow_paramstr(params_str)
    p = WordPOSPreprocessor()
    p.fit(X1=X1_train_test, X2=X2_train_test, Y=Y_train_test)
    model_config.adjust_params_follow_preprocessor(p)
    print(p.vocab_tag)

    # ----- Embedding loading -----
    w_embedding_path = '/home/s1610434/Documents/Data/Vector/{0}/{0}.word.{1}.txt'.format(model_config.embedding_name, model_config.word_embedding_size)
    pos_embedding_path = '/home/s1610434/Documents/Data/Vector/{0}/{0}.pos.{1}.txt'.format(model_config.embedding_name,model_config.pos_embedding_size)
    W_embedding = load_word_embeddings(p.vocab_word, w_embedding_path, model_config.word_embedding_size)
    POS_embedding = load_word_embeddings(p.pos_extractor.features_dict, pos_embedding_path, model_config.pos_embedding_size)

    print(W_embedding.shape)
    print(POS_embedding.shape)

    # for evaluation 2 tasks
    atepc_evaluator = ATEPCNewEvaluator()
    atepc_results = []


    for i_fold in range(10):
        hand_features_str = ",".join(hand_features) if hand_features is not None else "None"
        model_name = "{0}.{1}.{2}".format(keras_model_name,hand_features_str, params_str)
        model_name_ifold = model_name + "." + str(i_fold)

        p.load(os.path.join(save_path,model_name_ifold))
        X_test, Y_test = p.transform(X1=X1_test, X2=X2_test, Y=Y_test_origin)
        X_train, Y_train = p.transform(X1=X1_train_valid, X2=X2_train_valid, Y=Y_train_valid)
        X_valid, Y_valid = p.transform(X1=X1_train_valid, X2=X2_train_valid, Y=Y_train_valid)
        data = create_data_object(X_train, Y_train, X_valid, Y_valid, X_test, Y_test)

        sess = tf.Session()
        with sess.as_default():
            # tensorflow model
            model = AspectLSTM(config=model_config)
            model.saver.restore(sess, save_path=os.path.join(save_path,model_name_ifold))
            crf_transition_parameters = sess.run(model.crf_transition_parameters)
            # f1_valid, _, _, loss_valid = predict_step(sess, model, p, data, "valid", crf_transition_parameters)
            f1_test, ys_pred, ys_true, loss_test = predict_step(sess, model, p, data, "test", crf_transition_parameters)
            print("F1 test, ATEPC task: ", f1_test)
            f1, acc = atepc_evaluator.evaluate(ys_true, ys_pred, verbose=True)
            with open("logs/{0}.{1}.result.pickle".format(data_name,model_name_ifold), mode="wb") as f:
                pickle.dump((X1_test, ys_true, ys_pred, f1, acc), f)

            # write_result("logs/{0}.txt".format(model_name_ifold), X1_test, ys_true, ys_pred)
        tf.reset_default_graph()
        print("-----",i_fold,"-----")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-kr_name', type=str, default="W", help='keras_model_name')
    parser.add_argument('-task_name', type=str, default="ATEPC2", help='task_name')
    parser.add_argument('-data_name', type=str, default="restaurants16", help='data_name')
    parser.add_argument('-hand_features', type=str, default=None, help='hand_features')
    parser.add_argument('-params_str', type=str, default="w2v,150,050,200,20,0.0010,30,0.000,0", help='parameters')
    parser.add_argument('-attention_status', type=str, default="Without_Attention", help='attention_status')
    args = parser.parse_args()

    kr_names = ["WCH1", "WCH2", "WH1", "WH2", "WH", "WCH", "WC", "W", "WCP", "WP", "WPH", "WCPH", "WPD", "WPHD", "WPH1",
                "WPH2", "WPH3"]
    data_names = ["laptops", "restaurants", "restaurants15", "restaurants16"]

    if args.hand_features is None:
        hand_features = None
    else:
        hand_features = args.hand_features.split(",")

    params_str = args.params_str.strip()
    if args.kr_name in kr_names and args.data_name in data_names:
        train_model(keras_model_name=args.kr_name, data_name=args.data_name, hand_features=hand_features,
                    params_str=params_str, task_name=args.task_name, attention_status=args.attention_status)
    else:
        print("Wrong parameter, please choose params from these lists: ")
        print("-kr_name", kr_names)
        print("-data_name", data_names)


