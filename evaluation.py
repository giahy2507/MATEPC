from collections import defaultdict
import os
import xml.etree.ElementTree as ET
import re
import nltk
import copy
from shutil import copyfile
import subprocess
import numpy as np
from utils import get_aspecterm, collect_data_from_tsv, search_all



class ResultEvaluator(object):
    def __init__(self, name = ""):
        self.name = name

    def evaluate(self, pred_file, gold_file = ""):
        return

class ATEEvaluator(ResultEvaluator):
    def __init__(self, name="ATE Evaluation"):
        super(ATEEvaluator, self).__init__(name)

    @classmethod
    def get_entities(self, seq):
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
                for j in range(i + 1, len(seq)):
                    if seq[j].startswith('I') and types[j] == types[i]:
                        continue
                    break
                chunks.append(("ASP", i, j))
                i = j
            else:
                i += 1
        return chunks

    def count_correct_head_word(self, lab_chunks, lab_pred_chunks, two_sides = False):
        result = 0.
        for lab_chunk in lab_chunks:
            for lab_pred_chunk in lab_pred_chunks:
                lab_chunk_range = list(range(lab_chunk[1], lab_chunk[2]))
                lab_chunk_pred_range = list(range(lab_pred_chunk[1], lab_pred_chunk[2]))
                set_lab_chunk = set(lab_chunk_range) & set(lab_chunk_pred_range)
                if len(set_lab_chunk) > 0:
                    overlapping_head_idx = max(set(lab_chunk_range) & set(lab_chunk_pred_range))
                    if two_sides is True:
                        if overlapping_head_idx == lab_chunk_range[-1] or overlapping_head_idx == lab_chunk_pred_range[-1]:
                            result+=1
                    else:
                        if overlapping_head_idx == lab_chunk_range[-1]:
                            result+=1
                    # else:
                    #     print("ahihi")
        return result

    #TODO
    def f1_score(self, y_true, y_pred):
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
        correct_head_preds = 0.
        for lab, lab_pred in zip(y_true, y_pred):
            lab_chunks = set(self.get_entities(lab))
            lab_pred_chunks = set(self.get_entities(lab_pred))

            correct_head_preds += self.count_correct_head_word(lab_chunks, lab_pred_chunks)
            correct_preds += len(lab_chunks & lab_pred_chunks)
            total_preds += len(lab_pred_chunks)
            total_correct += len(lab_chunks)


        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

        p_head = correct_head_preds / total_preds if correct_head_preds > 0 else 0
        r_head = correct_head_preds / total_correct if correct_head_preds > 0 else 0
        f1_head = 2 * p_head * r_head / (p_head + r_head) if correct_head_preds > 0 else 0

        return p*100, r*100, f1*100, p_head*100, r_head*100, f1_head*100

    def evaluate(self, pred_file, gold_file=""):
        sents, labels, pred_labels = collect_data_from_tsv(pred_file)
        p, r, f1, p_head, r_head, f1_head  = self.f1_score(labels, pred_labels)
        return {"precision": "{:.2f}".format(p), "recall": "{:.2f}".format(r), "f1-score": "{:.2f}".format(f1),
                "precision-head": "{:.2f}".format(p_head), "recall-head": "{:.2f}".format(r_head), "f1-score-head": "{:.2f}".format(f1_head)}

class APCEvaluator(ResultEvaluator):
    def __init__(self, name="APC Evaluation"):
        super(APCEvaluator, self).__init__(name)

    def evaluate(self, pred_file, gold_file=""):
        senti_gold = []
        senti_pred = []
        sents, labels, pred_labels = collect_data_from_tsv(pred_file)
        for sent, label, pred in zip(sents, labels, pred_labels):
            sub_senti_gold, sub_senti_pred =  self.evaluate_sentence(sent, label, pred)
            senti_gold+=sub_senti_gold
            senti_pred+=sub_senti_pred
        senti_gold = np.array(senti_gold, dtype=np.int32)
        senti_pred = np.array(senti_pred, dtype=np.int32)
        accuracy = float(sum(senti_gold == senti_pred))/len(senti_gold)
        return {"accuracy": "{0:.2f}".format(accuracy*100)}

    def BIO2Sentiment(self, BIO_seq):
        term_mapping = {"POS": 1, "NEG": -1}

        sentiment_score = 0
        for term in BIO_seq:
            if term in term_mapping:
                sentiment_score += term_mapping[term]

        if sentiment_score > 0: return 2
        elif sentiment_score == 0: return 1
        else: return 0

    def evaluate_sentence(self, x, y, pred):
        senti_gold = []
        senti_pred = []
        i = 0
        y.append("O")
        while i < len(y):
            if y[i].split("-")[0] == "B":
                if y[i].split("-")[1] == "CON":
                    return [], []
                aspecterm = []
                gold_senti_seq = []
                pred_senti_seq = []
                aspecterm.append(x[i])
                gold_senti_seq.append(y[i].split("-")[-1])
                pred_senti_seq.append(pred[i].split("-")[-1])
                approx_pos = sum([len(word) + 1 for word in x[:i]])
                i += 1
                while y[i].split("-")[0] == "I" and i < len(y):
                    aspecterm.append(x[i])
                    gold_senti_seq.append(y[i].split("-")[-1])
                    pred_senti_seq.append(pred[i].split("-")[-1])
                    i += 1
                senti_gold.append(self.BIO2Sentiment(gold_senti_seq))
                senti_pred.append(self.BIO2Sentiment(pred_senti_seq))
            else:
                i += 1
        return senti_gold, senti_pred

class ContraintAPCEvaluator(APCEvaluator):
    def __init__(self, name="APC Evaluation under Constraint of ATE"):
        super(ContraintAPCEvaluator, self).__init__(name)

    def compare_2_lists(self, lista, listb):
        if len(lista) != len(listb):
            return False
        for i in range(len(lista)):
            if lista[i].split("-")[0] != listb[i].split("-")[0]:
                return False
        return True

    def get_pred_infor(self, y, pred):
        labels_gold = []
        labels_pred = []
        i = 0
        y.append("O")
        while i < len(y):
            if y[i].split("-")[0] == "B":
                aspecterm = []
                label_gold = []
                label_pred = []
                label_gold.append(y[i])
                label_pred.append(pred[i])
                i += 1
                while y[i].split("-")[0] == "I" and i < len(y):
                    label_gold.append(y[i])
                    label_pred.append(pred[i])
                    i += 1
                labels_gold.append(label_gold)
                labels_pred.append(label_pred)
            else:
                i += 1
        return labels_gold, labels_pred

    def check_conflict(self, y):
        for label in y:
            if label.split("-")[0] == "B":
                if label.split("-")[1] == "CON":
                    return True
        return False

    def _evaluate(self, labels, pred_labels):
        sentis_gold = []
        sentis_pred = []
        no_incorrect_ate = 0
        no_correct_ate = 0
        for label, pred in zip( labels, pred_labels):
            aspect_labels_gold, aspect_labels_pred = self.get_pred_infor(label, pred)
            for aspect_label_gold,  aspect_label_pred in zip( aspect_labels_gold, aspect_labels_pred):
                if aspect_label_gold[0] == "B-CON":
                    continue
                is_correct_ate = self.compare_2_lists(aspect_label_gold, aspect_label_pred)
                if is_correct_ate is False:
                    no_incorrect_ate += 1
                    continue
                else:
                    no_correct_ate +=1
                    senti_gold = self.BIO2Sentiment([label.split("-")[-1] for label in aspect_label_gold])
                    senti_pred = self.BIO2Sentiment([label.split("-")[-1] for label in aspect_label_pred])
                    sentis_gold.append(senti_gold)
                    sentis_pred.append(senti_pred)

        sentis_gold = np.array(sentis_gold, dtype=np.int32)
        sentis_pred = np.array(sentis_pred, dtype=np.int32)
        accuracy = float(sum(sentis_gold == sentis_pred))/len(sentis_gold) if len(sentis_gold) != 0 else 0
        return {"accuracy": "{0:.2f}".format(accuracy*100),
                "no_incorrect_ate": no_incorrect_ate,
                "no_correct_ate": no_correct_ate,
                "ate_correct_rate": "{0:.2f}".format((float(no_correct_ate)/(no_correct_ate+ no_incorrect_ate))*100),
                # "sentis_gold": sentis_gold,
                # "sentis_pred": sentis_pred,
                }

    def evaluate(self, pred_file, gold_file=""):
        sents, labels, pred_labels = collect_data_from_tsv(pred_file)
        return self._evaluate(labels, pred_labels)


class ATEPCEvaluator(ResultEvaluator):
    def __init__(self, name="ATEPC Evaluation"):
        super(ATEPCEvaluator, self).__init__(name)
        self.atesem_evaluator = ATESemEvaluator()
        self.ate_evaluator = ATEEvaluator()

        self.constraint_apc_evaluator = ContraintAPCEvaluator()
        self.ate_converter = ATEPC2ATEConverter()
        self.apc_converter = ATEPC2APCConverter()
        self.apc_evaluator = APCEvaluator()
        self.sem_converter = ATE2Semeval()

    def result_2_str(self, ate_result, apc_result, constraint_apc_result):
        result_str = ""
        result_str+="-- ATEPC without constraint --\n"
        result_str+="-- ATE --\n"
        result_str+="P: {0}, R: {1}, F1: {2}\n".format(ate_result["precision"], ate_result["recall"], ate_result["f1-score"])
        result_str+="-- APC --\n"
        result_str+="Acc: {0}\n".format(apc_result["accuracy"])
        result_str+="\n"
        result_str+="-- ATEPC with constraint --\n"
        result_str+="Acc: {0}, ATE correct rate: {1}\n".format(constraint_apc_result["accuracy"], constraint_apc_result["ate_correct_rate"])
        return result_str


    def evaluate(self, pred_file, gold_file = ""):
        if pred_file.find("laptops") != -1:
            ate_testphrase_file = os.path.join(os.getcwd(),"data/evaluation/laptops.ATETestPhrase.xml")
            ate_goldfile = os.path.join(os.getcwd(),"data/evaluation/laptops.ATEGold.xml")
        else:
            ate_testphrase_file = os.path.join(os.getcwd(), "data/evaluation/restaurants.ATETestPhrase.xml")
            ate_goldfile = os.path.join(os.getcwd(), "data/evaluation/restaurants.ATEGold.xml")

        try:
            # ATE Sem evaluation
            ate_filepath = self.ate_converter.convert(pred_file)
            pred_sem_file = self.sem_converter.convert(ate_filepath, semfile=ate_testphrase_file)
            atesem_result = self.atesem_evaluator.evaluate(pred_sem_file, gold_file=ate_goldfile)
            subprocess.call("rm {0}".format(pred_sem_file), shell=True)
        except:
            atesem_result = {"f1-score":0.0}

        # ATE evaluator
        ate_result = self.ate_evaluator.evaluate(pred_file)

        # APC evaluation
        apc_filepath = self.apc_converter.convert(pred_file)
        apc_result = self.apc_evaluator.evaluate(apc_filepath)

        # Constraint APC Evaluation
        constraint_apc_result = self.constraint_apc_evaluator.evaluate(pred_file)
        # print(self.result_2_str(ate_result, apc_result, constraint_apc_result))
        return atesem_result["f1-score"], ate_result["f1-score"], apc_result["accuracy"], constraint_apc_result["accuracy"]

class ATEPCNewEvaluator(object):
    def __init__(self, name="ATEPC New Evaluation"):
        self.name = name
        self.ate_evaluator = ATEEvaluator()
        self.constraint_apc_evaluator = ContraintAPCEvaluator()


    def evaluate(self, true_labels, pred_labels, verbose = False):
        true_labels_copy = copy.deepcopy(true_labels)
        pred_labels_copy = copy.deepcopy(pred_labels)
        # ATE evaluator
        p, r, f1, p_head, r_head, f1_head = self.ate_evaluator.f1_score(true_labels_copy, pred_labels_copy)
        ate_result = {"precision": "{:.2f}".format(p), "recall": "{:.2f}".format(r), "f1-score": "{:.2f}".format(f1),
                "precision-head": "{:.2f}".format(p_head), "recall-head": "{:.2f}".format(r_head),
                "f1-score-head": "{:.2f}".format(f1_head)}

        # Constraint APC Evaluation
        constraint_apc_result = self.constraint_apc_evaluator._evaluate(true_labels_copy, pred_labels_copy)
        if verbose:
            print(constraint_apc_result)
            print(ate_result)
        return ate_result["f1-score"], constraint_apc_result["accuracy"]


if __name__ == "__main__":

    file_data = "data/W.NEGAT,BING,SWN,NAMEL,DEPENCY,HEADVOTE.w2v,200,200,100,100,050,0.5,20,0.0010,1,1,0.45.0.restaurants15.ATEPC2.test.pred.tsv"
    atepc_evaluator = ATEPCEvaluator()
    aaa = atepc_evaluator.evaluate(file_data)
