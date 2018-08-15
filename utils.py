import os
import numpy as np
import re
from sklearn.model_selection import KFold

def dense_to_one_hot(labels_dense, num_classes, nlevels=1):
    """Convert class labels from scalars to one-hot vectors."""
    if nlevels == 1:
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes), dtype=np.int32)
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot
    elif nlevels == 2:
        # assume that labels_dense has same column length
        num_labels = labels_dense.shape[0]
        num_length = labels_dense.shape[1]
        labels_one_hot = np.zeros((num_labels, num_length, num_classes), dtype=np.int32)
        layer_idx = np.arange(num_labels).reshape(num_labels, 1)
        # this index selects each component separately
        component_idx = np.tile(np.arange(num_length), (num_labels, 1))
        # then we use `a` to select indices according to category label
        labels_one_hot[layer_idx, component_idx, labels_dense] = 1
        return labels_one_hot
    else:
        raise ValueError('nlevels can take 1 or 2, not take {}.'.format(nlevels))

def collect_data_infor_from_tsv(tsvfile, keep_conflict = False):
    if os.path.isfile(tsvfile) == False:
        raise ("[!] Data %s not found" % tsvfile)

    conflict_flag = False
    count_conflict=  0
    # Collect sentences in tsv file
    sents,  poses, dep_idxs, dep_relations, labels, preds = [], [], [], [], [], []
    with open(tsvfile) as f:
        sent, pos, dep_idx, dep_relation, label, pred = [], [], [], [], [], []
        for line in f:
            line = line.rstrip()
            if len(line) == 0 or line.startswith('-DOCSTART-'):
                if len(sent) != 0:
                    if keep_conflict is False and conflict_flag is True:
                        count_conflict+=1
                        conflict_flag = False
                    else:
                        sents.append(sent)
                        poses.append(pos)
                        dep_idxs.append(dep_idx)
                        dep_relations.append(dep_relation)
                        labels.append(label)
                        preds.append(pred)
                    sent, pos, dep_idx, dep_relation, label, pred = [], [], [], [], [], []
            else:
                tokens = line.split('\t')
                sent.append(tokens[0])
                pos.append(tokens[1])
                dep_idx.append(tokens[2])
                dep_relation.append(tokens[3])
                label.append(tokens[4])
                if tokens[4] == "B-CON":
                    conflict_flag = True
                if len(tokens) == 6:
                    pred.append(tokens[5])
                else:
                    pred.append("")
    return np.asarray(sents),  np.asarray(poses), np.asarray(dep_idxs), np.asarray(dep_relations), np.asarray(labels), np.asarray(preds)

def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple.
        pad_tok: the char to pad with.

    Returns:
        a list of list where each sublist has same length.
    """
    sequence_padded = []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        if max_length < len(seq):
            raise Exception("len(seq) > max_length")
    return sequence_padded

def pad_sequences(sequences, pad_tok, max_length = None):
    """
    Args:
        sequences: a generator of list or tuple.
        pad_tok: the char to pad with.

    Returns:
        a list of list where each sublist has same length.
    """
    #TODO
    if max_length is None:
        max_length = len(max(sequences, key=len))
    sequence_padded= _pad_sequences(sequences, pad_tok, max_length)
    return np.asarray(sequence_padded)


def load_word_embeddings(vocab, filename, dim):
    """Loads vectors in numpy array.

    Args:
        vocab (): dictionary vocab[word] = index.
        filename (str): a path to a glove file.
        dim (int): dimension of embeddings.

    Returns:
        numpy array: an array of word embeddings.
    """
    print("Load" + filename)
    embeddings = 0.2 * np.random.uniform(-1.0, 1.0, (len(vocab), dim))
    if os.path.isfile(filename):
        with open(filename) as f:
            for line in f:
                line = line.strip().split(' ')
                word = line[0]
                embedding = [float(x) for x in line[1:dim+1]]
                if word in vocab:
                    word_idx = vocab[word]
                    embeddings[word_idx] = np.asarray(embedding)
    else:
        print("Init embeding randomly")
    embeddings[vocab['<PAD>']] = np.array([0]*dim, dtype=embeddings.dtype)
    return embeddings

def collect_data_from_tsv(tsvfile):
    if os.path.isfile(tsvfile) == False:
        raise ("[!] Data %s not found" % tsvfile)
    # Collect sentences in tsv file
    sents, labels, pred_labels = [], [], []
    with open(tsvfile) as f:
        words, tags, preds = [], [], []
        for line in f:
            line = line.rstrip()
            if len(line) == 0 or line.startswith('-DOCSTART-'):
                if len(words) != 0:
                    sents.append(words)
                    labels.append(tags)
                    pred_labels.append(preds)
                    words, tags, preds = [], [], []
            else:
                tokens = line.split('\t')
                word = tokens[0]
                if word == "''":
                    word = "\""
                if word=="``":
                    word = "\""
                words.append(word)
                tags.append(tokens[1])
                if len(tokens) == 3:
                    preds.append(tokens[2])
                else:
                    preds.append("")
    return np.asarray(sents), np.asarray(labels), np.asarray(pred_labels)

def search_all(pattern, string):
    result = []
    finded = re.finditer(pattern=pattern, string=string)
    for find in finded:
        result.append(find.regs[0])
    return result

def get_aspecterm(x, y):
    result = []
    i = 0
    y.append("O")
    while i < len(y):
        if y[i].split("-")[0] == "B":
            aspecterm = []
            term = x[i].lower()
            aspecterm.append(term)
            approx_pos = sum([len(word) + 1 for word in x[:i]])
            i += 1
            while y[i].split("-")[0] == "I" and i < len(y):
                term = x[i].lower()
                aspecterm.append(term)
                i += 1
            result.append({"aspect_term": aspecterm, "approx_pos": approx_pos})
        else:
            i += 1
    return result

def write_file(file_no, list_write):
    with open("runparams{0}.sh".format(file_no), mode="w") as f:
        f.write("#!/usr/bin/env bash\n")
        for line in list_write:
            f.write(line+"\n")

if __name__ == '__main__':
    list_write = []
    times = 10
    list_features = ["None"]
    for task_name in ["ATE", "ATEPC2"]:
        for keras_model_name in ["W"]:
            for data_name in ["laptops", "restaurants"]:
                for eb_type in ["w2v"]:
                    if data_name == "laptops":
                        w_e = "150"
                        nwh = "200"
                        p_e = "050"
                    elif data_name == "restaurants" or data_name == "restaurants15":
                        w_e = "150"
                        nwh = "200"
                        p_e = "050"
                    for bs in [20]:
                        for lr in ["0.0010"]:
                            for hand_feature_str in list_features:
                                for i in range(times):
                                    script = "python " + "train.py -task_name {3} -kr_name {0} -hand_features {1} -data_name {2} -params_str ".format(keras_model_name, hand_feature_str ,data_name, task_name) + "w2v,150,050,200,20,0.0010,30,0.000,0"
                                    list_write.append(script)

    list_write = np.asarray(list_write)
    kf = KFold(n_splits=4)
    file_no = 0
    for train_index, valid_index in kf.split(list_write):
        f_fold_fn = list_write[valid_index]
        write_file(file_no=file_no, list_write=f_fold_fn)
        file_no+=1
