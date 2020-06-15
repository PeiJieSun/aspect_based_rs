from bleu import *
from rouge import rouge

import numpy as np

PAD = 0

def tensorToScalar(tensor):
    return tensor.cpu().detach().numpy()


def evaluate(test_dataset, test_batch_sampler, model):
    bleu_score = []
    bleu_list_1, bleu_list_2, bleu_list_3, bleu_list_4 = [], [], [], []
    rouge_1_list, rouge_2_list, rouge_L_list = [], [], []

    hyp_ref_list = []

    for batch_idx_list in test_batch_sampler:
        user, item, review_input, real_review = test_dataset.get_batch(batch_idx_list)
        sample_idx_list = model._sample_text_by_top_one(user, item, review_input)
        
        for record_idx, hyp in enumerate(tensorToScalar(sample_idx_list)):
            for clip_idx, word in enumerate(hyp):
                if word == PAD:
                    break
            hyp = hyp[:clip_idx]
            hyp_ref_list.append([hyp.tolist(), [tensorToScalar(real_review[record_idx])]])

    
    for [hyp, ref] in hyp_ref_list:
        #import pdb; pdb.set_trace()
        try:
            bleu_score = compute_bleu([hyp], [ref])
            bleu_list_1.append(bleu_score[1])
            bleu_list_2.append(bleu_score[2])
            bleu_list_3.append(bleu_score[3])
            bleu_list_4.append(bleu_score[4])

            rouge_score = rouge([hyp], ref)
            rouge_1_list.append(rouge_score[0])
            rouge_2_list.append(rouge_score[1])
            rouge_L_list.append(rouge_score[2])
        except:
            pass

    print('bleu_1:%.4f' % np.mean(bleu_list_1))
    print('bleu_2:%.4f' % np.mean(bleu_list_2))
    print('bleu_3:%.4f' % np.mean(bleu_list_3))
    print('bleu_4:%.4f' % np.mean(bleu_list_4))
    print('rouge_1_f:%.4f' % np.mean(rouge_1_list))
    print('rouge_2_f:%.4f' % np.mean(rouge_2_list))
    print('rouge_L_f:%.4f' % np.mean(rouge_L_list))

    return np.mean(bleu_list_4), np.mean(rouge_L_list)
'''
def evaluate(test_dataset, test_batch_sampler, model):
    bleu_score = []
    bleu_list_1, bleu_list_2, bleu_list_3, bleu_list_4 = [], [], [], []
    rouge_1_list, rouge_2_list, rouge_L_list = [], [], []
    out_probit = []; target = []
    for batch_idx_list in test_batch_sampler:
        user, item, review_input, real_review = \
            test_dataset.get_batch(batch_idx_list)
        sample_idx_list = model._sample_text_by_top_one(user, item, review_input)
        ref = tensorToScalar(real_review).tolist()
        try:
            bleu_score = compute_bleu([sample_idx_list], [ref])
            bleu_list_1.append(bleu_score[1])
            bleu_list_2.append(bleu_score[2])
            bleu_list_3.append(bleu_score[3])
            bleu_list_4.append(bleu_score[4])

            rouge_score = rouge([sample_idx_list], ref)
            rouge_1_list.append(rouge_score[0])
            rouge_2_list.append(rouge_score[1])
            rouge_L_list.append(rouge_score[2])
        except:
            pass

    print('bleu_1:%.4f' % np.mean(bleu_list_1))
    print('bleu_2:%.4f' % np.mean(bleu_list_2))
    print('bleu_3:%.4f' % np.mean(bleu_list_3))
    print('bleu_4:%.4f' % np.mean(bleu_list_4))
    print('rouge_1_f:%.4f' % np.mean(rouge_1_list))
    print('rouge_2_f:%.4f' % np.mean(rouge_2_list))
    print('rouge_L_f:%.4f' % np.mean(rouge_L_list))

    return np.mean(bleu_list_4), np.mean(rouge_L_list)
'''