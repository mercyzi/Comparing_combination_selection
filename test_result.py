import pandas as pd
from conf import *
import json
import os
import pickle
import numpy as np
import torch
import random
import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix

def get_dirichlet_params(acc, strength, n_cls):
    beta = 0.1
    alpha = beta * (n_cls - 1) * acc / (1. - acc)

    alpha *= strength
    beta *= strength

    alpha += 1
    beta += 1

    return alpha, beta

def write_data(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        print('dumping json object to {}.'.format(path))
def load_json(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class ModelWithTemperature(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits
        self.temperature = torch.nn.Parameter(torch.ones(1)* 1.5)

    def forward(self):
        return torch.softmax(self.logits / self.temperature, dim=-1)
        
stat_rate = 0.9
for nums_class in range(3,17,1):
    for ep in [1,10]:
        for noise_level in [80,95,110,125]:
            saved_json = f'res/all_limit_class/{nums_class}_class/ep{ep}_noise_{noise_level}.json'
            save_dir = os.path.dirname(saved_json)
            os.makedirs(save_dir, exist_ok=True)
            res_json = {}
            print('noise_level: ',noise_level)
            print('-'*20)
            all_record_h_acc = []
            all_record_m_acc = []
            all_record_hm_acc_and = []
            all_record_hm_he_and = []
            all_record_hm_acc_or = []
            all_record_hm_he_or = []
            all_record_m_conf = []
            all_record_h_conf = []
            all_record_mat_m_conf = []
            all_record_prob_h_conf = []                
            all_record_hm_4_and = []
            all_record_hm_4_or = []
            for model_name in ['alexnet','resnet152','googlenet','densenet161','vgg19']:
                df = pd.read_csv("Behavioral Data/Preprocessed/human_only_classification_6per_img_preprocessed.csv")
                human_data = []
                for index, row in df.iterrows():
                    sample_dict = row.to_dict()
                    if sample_dict['noise_level'] == noise_level:
                        human_data.append(sample_dict)
                machine_data = []
                df = pd.read_csv(f"Machine Classifier Predictions/hai_epoch{ep}_model_preds_max_normalized.csv") # hai_epoch10_model_preds_max_normalized # hai_epoch01_model_preds_max_normalized
                for index, row in df.iterrows():
                    sample_dict = row.to_dict()
                    if sample_dict['noise_level'] == noise_level and sample_dict['model_name'] == model_name:
                        machine_data.append(sample_dict)
                target_classes = ['knife', 'keyboard', 'elephant', 'bicycle', 'airplane',
                                'clock', 'oven', 'chair', 'bear', 'boat', 'cat', 'bottle',
                                'truck', 'car', 'bird', 'dog']
                class_to_int = {cls_name: idx for idx, cls_name in enumerate(target_classes)}
                hm_data = []
                for m_sample in machine_data:
                    img_name = m_sample['image_name']
                    for h_sample in human_data:
                        if img_name == h_sample['image_name']:
                            h_pred = h_sample['participant_classification']
                            h_id = h_sample['worker_id']
                            ai_prob = [m_sample[class_name] for class_name in target_classes if class_name in m_sample]
                            ai_pred = m_sample['model_pred']
                            label = h_sample['image_category']
                            new_hm_sample = {'image_name':img_name, 'id':h_sample['id'],'h_id': h_id,'label': class_to_int[label], 'h_pred': class_to_int[h_pred],'ai_pred': class_to_int[ai_pred],'ai_prob': ai_prob,'h_total_acc': h_sample['total_accuracy']}
                            hm_data.append(new_hm_sample)
                logits_val = torch.log(torch.tensor([sample['ai_prob'] for sample in hm_data]))  
                labels_val = torch.tensor([sample['label'] for sample in hm_data])  
                logits_val = torch.nan_to_num(logits_val, nan=0.0)
                loss_threshold = 1e-6  
                Scaling_model = ModelWithTemperature(logits_val)
                nll_criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam([Scaling_model.temperature], lr=0.001)
                for _ in range(1000):
                    optimizer.zero_grad()
                    loss = nll_criterion(Scaling_model.logits / Scaling_model.temperature, labels_val)
                    loss.backward()
                    optimizer.step()
                    if loss.item() < loss_threshold:
                        print(f"epoch={epoch}, loss={loss.item():.8f}")
                        break
                T = Scaling_model.temperature.item()
                multi_hm_data = {'1': [], '2': [], '3': [], '4': [], '5': [], '6': []}
                for m_sample in machine_data:
                    img_name = m_sample['image_name']
                    matched_hms = [hm for hm in hm_data if hm['image_name'] == img_name]
                    sorted_hms = sorted(matched_hms, key=lambda x: x['h_total_acc'], reverse=True)
                    for i, hm in enumerate(sorted_hms[:6]):
                        key = str(i + 1)
                        multi_hm_data[key].append(hm)
                merged_multi_hm_data = {
                    '1': multi_hm_data['1'],
                    '2': multi_hm_data['2'],
                    '3': multi_hm_data['3'],
                    '4': multi_hm_data['4'],
                    '5': multi_hm_data['5'],
                    '6': multi_hm_data['6']
                }
                for human_id in merged_multi_hm_data.keys():
                    solo_record_h_acc = []
                    solo_record_m_acc = []
                    solo_record_hm_acc_and = []
                    solo_record_hm_he_and = []
                    solo_record_hm_acc_or = []
                    solo_record_hm_he_or = []
                    solo_record_m_conf = []
                    solo_record_h_conf = []
                    solo_record_mat_m_conf = []
                    solo_record_prob_h_conf = []                
                    solo_record_hm_4_and = []
                    solo_record_hm_4_or = []
                    solo_hm_nums = len(multi_hm_data[human_id])
                    solo_hm_samples = multi_hm_data[human_id]
                    hm_4_and = [0,0,0,0]
                    hm_4_or = [0,0,0,0]
                    hm_4_all = [0,0,0,0]
                    for rand_seed in range(1,301,1):
                        set_seed(rand_seed+1)
                        chosen_classes = np.random.choice(range(16), size=nums_class, replace=False)
                        class_map = {old: new for new, old in enumerate(chosen_classes)}
                        filtered_samples = []
                        for s in solo_hm_samples:
                            if s['label'] in chosen_classes and s['h_pred'] in chosen_classes and s['ai_pred'] in chosen_classes:
                                new_s = s.copy()
                                new_s['label'] = class_map[s['label']]
                                new_s['h_pred'] = class_map[s['h_pred']]
                                new_s['ai_pred'] = class_map[s['ai_pred']]
                                ai_prob_new = [s['ai_prob'][old] for old in chosen_classes]
                                ai_prob_new = np.array(ai_prob_new) / np.sum(ai_prob_new)
                                new_s['ai_prob'] = ai_prob_new
                                filtered_samples.append(new_s)
                        indices = np.random.choice(len(filtered_samples), size=len(filtered_samples), replace=False)
                        stat_indices = indices[:int(len(filtered_samples)*stat_rate)]
                        test_indices = indices[int(len(filtered_samples)*stat_rate):]
                        stat_solo_hm_samples = [filtered_samples[i] for i in stat_indices]
                        test_solo_hm_samples = [filtered_samples[i] for i in test_indices]
                        stat_pred = [sample['h_pred'] for sample in stat_solo_hm_samples]
                        stat_true = [sample['label'] for sample in stat_solo_hm_samples]
                        diag_acc = 0.75
                        strength = 1.0
                        alpha, beta = get_dirichlet_params(diag_acc, strength, nums_class)
                        prior_matr = np.eye(nums_class) * alpha + (np.ones(nums_class) - np.eye(nums_class)) * beta
                        posterior_matr = 1. * confusion_matrix(np.array(stat_true), np.array(stat_pred), labels=np.arange(nums_class))
                        posterior_matr += prior_matr
                        posterior_matr = posterior_matr.T
                        posterior_matr = (posterior_matr) / (np.sum(posterior_matr, axis=0, keepdims=True))
                        posterior_matr = torch.tensor(posterior_matr)
                        all_max_probs = []
                        h_he_or = 0
                        for sample in test_solo_hm_samples:
                            ai_prob = sample['ai_prob']
                            logits = torch.log(torch.tensor(ai_prob))
                            logits = torch.nan_to_num(logits, nan=0.0)
                            T = Scaling_model.temperature.item()
                            ai_prob = torch.softmax(logits / T, dim=-1)
                            max_prob, _ = torch.max(ai_prob, dim=0)
                            label = sample['label']
                            h_pred = sample['h_pred']
                            ai_pred = sample['ai_pred']
                            diag = torch.diag(posterior_matr)
                            p_correct_human = torch.sum(ai_prob * diag)
                            # deferral: compute human participation
                            if max_prob.item() <= p_correct_human.item():
                                h_he_or += 1  
                            all_max_probs.append(max_prob.item())
                        dymaic_he = -int(h_he_or*(1/4)*1)
                        n = len(all_max_probs)
                        min_dymaic_he = - (h_he_or - 1)
                        max_dymaic_he = n - h_he_or
                        dymaic_he = max(min(dymaic_he, max_dymaic_he - 1), min_dymaic_he)
                        if h_he_or > 0:
                            sorted_probs = np.sort(all_max_probs)
                            threshold = sorted_probs[h_he_or - 1 + dymaic_he]  
                        else:
                            threshold = 0.0 
                        threshold = 1.0
                        # test hm
                        h_true = 0
                        m_true = 0
                        hm_true_and = 0
                        h_he_and = 0
                        hm_true_or = 0
                        h_he_or = 0
                        m_conf = [0,0,0,0]
                        h_conf = [0,0,0,0]
                        mat_m_conf = [0,0,0,0]
                        prob_h_conf = [0,0,0,0]
                        for id_sample, sample in enumerate(test_solo_hm_samples):
                            label = sample['label']
                            h_pred = sample['h_pred']
                            ai_pred = sample['ai_pred']
                            ai_prob = sample['ai_prob']
                            logits = torch.log(torch.tensor(ai_prob))
                            logits = torch.nan_to_num(logits, nan=0.0)
                            T = Scaling_model.temperature.item()
                            ai_prob = torch.softmax(logits / T, dim=-1)
                            max_prob, max_class = torch.max(ai_prob, dim=0)
                            if label == h_pred:
                                h_true += 1
                            if label == ai_pred:
                                m_true += 1
                            diag = torch.diag(posterior_matr) 
                            p_correct_human = torch.sum(ai_prob * diag)
                            if label == h_pred and label == ai_pred:
                                hm_4_all[0] += 1
                            elif label == h_pred and label != ai_pred:
                                hm_4_all[1] += 1
                            elif label != h_pred and label == ai_pred:
                                hm_4_all[2] += 1
                            else:
                                hm_4_all[3] += 1
                            # combine
                            if max_prob.item() <= threshold:
                                hm_pred_and = (ai_prob * posterior_matr[h_pred]).argmax(dim=-1).item()
                                h_he_and += 1
                            else:
                                hm_pred_and = ai_pred
                            if label == hm_pred_and:
                                hm_true_and += 1
                                if label == h_pred and label == ai_pred:
                                    hm_4_and[0] += 1
                                elif label == h_pred and label != ai_pred:
                                    hm_4_and[1] += 1
                                elif label != h_pred and label == ai_pred:
                                    hm_4_and[2] += 1
                                else:
                                    hm_4_and[3] += 1
                            # select
                            if max_prob.item() > p_correct_human.item(): 
                                hm_pred_or = ai_pred 
                            else: 
                                hm_pred_or = h_pred 
                                h_he_or += 1
                            if label == hm_pred_or:
                                hm_true_or += 1
                                if label == h_pred and label == ai_pred:
                                    hm_4_or[0] += 1
                                elif label == h_pred and label != ai_pred:
                                    hm_4_or[1] += 1
                                elif label != h_pred and label == ai_pred:
                                    hm_4_or[2] += 1
                                else:
                                    hm_4_or[3] += 1
                        solo_record_h_acc.append(h_true/len(test_solo_hm_samples))
                        solo_record_m_acc.append(m_true/len(test_solo_hm_samples))
                        solo_record_hm_acc_and.append(hm_true_and/len(test_solo_hm_samples))
                        solo_record_hm_he_and.append(h_he_and/len(test_solo_hm_samples))
                        solo_record_hm_acc_or.append(hm_true_or/len(test_solo_hm_samples))
                        solo_record_hm_he_or.append(h_he_or/len(test_solo_hm_samples))
                    solo_record_hm_4_and = [i/j for i, j in zip(hm_4_and, hm_4_all)]
                    solo_record_hm_4_or = [i/j for i, j in zip(hm_4_or, hm_4_all)]
                    all_record_h_acc.append(solo_record_h_acc)
                    all_record_m_acc.append(solo_record_m_acc)
                    all_record_hm_acc_and.append(solo_record_hm_acc_and)
                    all_record_hm_he_and.append(solo_record_hm_he_and)
                    all_record_hm_acc_or.append(solo_record_hm_acc_or)
                    all_record_hm_he_or.append(solo_record_hm_he_or)
                    all_record_m_conf.append(solo_record_m_conf)
                    all_record_h_conf.append(solo_record_h_conf)
                    all_record_hm_4_and.append(solo_record_hm_4_and)
                    all_record_hm_4_or.append(solo_record_hm_4_or)
                    all_record_mat_m_conf.append(solo_record_mat_m_conf)
                    all_record_prob_h_conf.append(solo_record_prob_h_conf)
            res_json['H_acc'] = all_record_h_acc
            res_json['M_acc'] = all_record_m_acc
            res_json['HM_acc_and'] = all_record_hm_acc_and
            res_json['HM_acc_or'] = all_record_hm_acc_or
            res_json['HM_he_and'] = all_record_hm_he_and
            res_json['HM_he_or'] = all_record_hm_he_or
            res_json['M_conf'] = all_record_m_conf
            res_json['H_conf'] = all_record_h_conf
            res_json['HM_4_and'] = all_record_hm_4_and
            res_json['HM_4_or'] = all_record_hm_4_or
            res_json['Mat_M_conf'] = all_record_mat_m_conf
            res_json['Prob_H_conf'] = all_record_prob_h_conf
            write_data(res_json,saved_json)
            print('HM_acc_and: ',np.mean(all_record_hm_acc_and, axis=1))
            print('HM_acc_or: ',np.mean(all_record_hm_acc_or, axis=1))
            print('HM_he_and: ',np.mean(all_record_hm_he_and, axis=1))
            print('HM_he_or: ',np.mean(all_record_hm_he_or, axis=1))
            print('HM_4_and: ',np.mean(all_record_hm_4_and, axis=(0)))
            print('HM_4_or: ',np.mean(all_record_hm_4_or, axis=(0)))
