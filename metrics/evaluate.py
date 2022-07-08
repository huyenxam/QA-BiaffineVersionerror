import json
from metrics.f1_score import f1_score
from metrics.exact_match_score import exact_match_score
from dataloader import *

def evaluate(predictions, max_char_len, max_seq_length, stride, mode):
    list_sample = []

    if mode == 'dev':
        path = './DataNew/dev_ViQuAD.json'
    elif mode == 'test':
        path = './DataNew/test_ViQuAD.json'
    else:
        raise Exception("Only dev and test dataset available")
        
    f1 = exact_match = total = 0
    i = 0
    list_sample = []
    with open(path, 'r', encoding='utf8') as f:
        list_sample = json.load(f)

    for sample in list_sample: 
        context = sample['context'].split(' ')
        question = sample['question'].split(' ')
        text_context = context

        max_seq = max_seq_length - len(question) - 2 
        if len(context) <= max_seq:
            sent = ['cls'] + question + ['sep'] +  context
            f1_idx = [0]
            extract_match_idx = [0]

            start_pre = int(predictions[i][1])
            end_pre = int(predictions[i][2])
            label_prediction = " ".join(sent[start_pre:end_pre+1])

            labels = sample['label']
            for lb in labels:
                start = lb[1]
                end = lb[2]
                ground_truth = " ".join(text_context[start:end+1])
                # ground_truth = lb[3]
                # print(ground_truth)
                # print(label_prediction)
                f1_idx.append(f1_score(label_prediction, ground_truth))
                extract_match_idx.append(exact_match_score(label_prediction, ground_truth))
            i += 1
        else:
            score_max = 0
            label_prediction = ""
            while(len(context) > max_seq):                     
                ctx = context[:max_seq]
                sent = ['cls'] + question + ['sep'] +  ctx
                context = context[stride:]
                score = predictions[i][3]
                start_pre = int(predictions[i][1])
                end_pre = int(predictions[i][2])
                if start_pre != 0 and end_pre != 0:
                    if score > score_max:
                        score_max = score
                        # start_pre = int(predictions[i][1])
                        # end_pre = int(predictions[i][2])
                        label_prediction = " ".join(sent[start_pre:end_pre+1])
                i += 1
            labels = sample['label']
            f1_idx = [0]
            extract_match_idx = [0]
            for lb in labels:
                start = lb[1]
                end = lb[2]
                ground_truth = " ".join(text_context[start:end+1])
                # ground_truth = lb[3] 
                # print(ground_truth)
                # print(label_prediction)
                f1_idx.append(f1_score(label_prediction, ground_truth))
                extract_match_idx.append(exact_match_score(label_prediction, ground_truth))
        f1 += max(f1_idx)
        exact_match += max(extract_match_idx)
        total += 1
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    
    return exact_match, f1

    # list_sample = InputSample(path=path, max_char_len=max_char_len, max_seq_length=max_seq_length, stride=stride).get_sample()
    # for i, sample in enumerate(list_sample):

    #     context = sample['context']
    #     question = sample['question']
    #     sentence = ['cls'] + question + ['sep'] + context

    #     labels = sample['label_idx']

    #     f1_idx = [0]
    #     extract_match_idx = [0]
    #     for lb in labels:

    #         start = lb[1]
    #         end = lb[2]
    #         ground_truth = " ".join(sentence[start:end+1])
            
    #         start_pre = int(predictions[i][1])
    #         end_pre = int(predictions[i][2])
    #         label_prediction = " ".join(sentence[start_pre:end_pre+1])
    #         if start == 0 and end == 0:
    #             break
    #         f1_idx.append(f1_score(label_prediction, ground_truth))
    #         extract_match_idx.append(exact_match_score(label_prediction, ground_truth))
    #         # print(ground_truth)
    #         # print(label_prediction)
            
    #     f1 += max(f1_idx)
    #     exact_match += max(extract_match_idx)
    #     total += 1
        

    # exact_match = 100.0 * exact_match / total
    # f1 = 100.0 * f1 / total
    
    # return exact_match, f1