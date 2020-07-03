from bleu import *
from rouge import rouge

import numpy as np

PAD = 0

def tensorToScalar(tensor):
    return tensor.cpu().detach().numpy()

def evaluate(test_dataset, test_batch_sampler, model, review_aspect_mask):
    model.eval()
    
    bleu_score = []
    bleu_list_1, bleu_list_2, bleu_list_3, bleu_list_4 = [], [], [], []
    rouge_1_list, rouge_2_list, rouge_L_list = [], [], []

    hyp_ref_list = []

    for batch_idx_list in test_batch_sampler:
        
        #user, item, review_input, summary, real_review = \
        #    test_dataset.get_batch(batch_idx_list)
        #sample_idx_list = \
        #    model._sample_text_by_top_one(user, item, summary, review_input, \
        #    review_aspect, review_aspect_mask)
    
        user, item, review_input, summary, real_review = test_dataset.get_batch(batch_idx_list)
        sample_idx_list = model._sample_text_by_top_one(user, item, review_input, review_aspect_mask)

        #import pdb; pdb.set_trace()
        #for record_idx, hyp in enumerate(tensorToScalar(sample_idx_list)):
        for record_idx, hyp in enumerate(sample_idx_list):
            hyp = tensorToScalar(hyp).tolist()
            for clip_idx, word in enumerate(hyp):
                if word == PAD:
                    # if current word is the last word of hyp
                    if clip_idx + 1 == len(hyp):
                        clip_idx = clip_idx - 1
                        break
                    # if next word also the PAD
                    elif hyp[clip_idx + 1] == PAD:
                        clip_idx = clip_idx - 1
                        break
            hyp = hyp[:clip_idx+1]

            #import pdb; pdb.set_trace()
            ref = tensorToScalar(real_review[record_idx]).tolist()
            for clip_idx, word in enumerate(ref):
                if word == PAD:
                    # if current word is the last word of ref
                    if clip_idx + 1 == len(ref):
                        clip_idx = clip_idx - 1
                        break
                    # if next word also the PAD
                    elif ref[clip_idx + 1] == PAD:
                        clip_idx = clip_idx - 1
                        break
            ref = ref[:clip_idx+1]

            if len(ref) != 0:
                hyp_ref_list.append([hyp, [ref]])

    #import pdb; pdb.set_trace()
    for record_idx, [hyp, ref] in enumerate(hyp_ref_list):
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

    #import pdb; pdb.set_trace()
    print('bleu_1:%.4f' % np.mean(bleu_list_1))
    print('bleu_2:%.4f' % np.mean(bleu_list_2))
    print('bleu_3:%.4f' % np.mean(bleu_list_3))
    print('bleu_4:%.4f' % np.mean(bleu_list_4))
    print('rouge_1_f:%.4f' % np.mean(rouge_1_list))
    print('rouge_2_f:%.4f' % np.mean(rouge_2_list))
    print('rouge_L_f:%.4f' % np.mean(rouge_L_list))

    return np.mean(bleu_list_4), np.mean(rouge_L_list)   