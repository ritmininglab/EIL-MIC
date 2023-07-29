
from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize
from nltk.translate import meteor
from rouge import Rouge
import numpy as np

def calculate_bleu(candidate, references):
    if len(references)==0 or len(references[0])==0:
        return [1,1,1,1,1]
    reference = [word_tokenize(ref) for ref in references]
    candi = word_tokenize(candidate)
    score1 = sentence_bleu(reference, candi, weights=[1,0,0,0])
    score2 = sentence_bleu(reference, candi, weights=[0,1,0,0])
    score3 = sentence_bleu(reference, candi, weights=[0,0,1,0])
    score4 = sentence_bleu(reference, candi, weights=[0,0,0,1])
    score = np.mean([score1, score2, score3, score4])
    return [score, score1, score2, score3, score4]


def calculate_meteor(candidate, references):
    if len(references)==0 or len(references[0])==0:
        return 1
    reference = [word_tokenize(ref) for ref in references]
    candi = word_tokenize(candidate)
    meteor_score = round(meteor(reference,candi), 4)
    return meteor_score


rouge = Rouge()

def calculate_rouge(candidate, references):
    score_l = 0
    if len(references)==0 or len(references[0])==0:
        return 1
    for ref in references:
        scores = rouge.get_scores(candidate, ref)
        score_l = max(score_l, scores[0]["rouge-l"]["f"])
    return score_l