import torch
import json
import numpy as np
from torch.utils.data import Dataset


class InputSample(object):
    def __init__(self, path, max_char_len=None, max_seq_length=None, stride=None):
        self.stride = stride                        # Số từ giữ lại sau mỗi lần cắt
        self.max_char_len = max_char_len            # Độ dài tối đa đầu vào của CharCNN
        self.max_seq_length = max_seq_length        # Độ dài tối của context
        self.list_sample = []
        with open(path, 'r', encoding='utf8') as f:
            self.list_sample = json.load(f)

    def get_character(self, word, max_char_len):
        word_seq = []
        for j in range(max_char_len):
            try:
                char = word[j]
            except:
                char = 'PAD'
            word_seq.append(char)
        return word_seq

    def get_sample(self):
        l_sample = []
        for i, sample in enumerate(self.list_sample):               # Lặp qua từng sample
            context = sample['context'].split(' ')                  # Tách context thành từng từ
            question = sample['question'].split(' ')                # Tách question thành từng từ
            sample['question'] = question                           # Lưu sample['question] 
            
            max_seq = self.max_seq_length - len(question) - 2       # 3: [cls] , [sep], [sep]
            if len(context) <= max_seq:                             # Nếu độ dài context nhỏ hơn max_sequence
                sent = question + context   
                sample['context'] = context

                char_seq = []
                for word in sent:
                    character = self.get_character(word, self.max_char_len)
                    char_seq.append(character)
                sample['char_sequence'] = char_seq

                label = sample['label']
                label_idxs = []

                se = ['cls'] + question + ['sep'] +  context
                for lb in label:
                    ans_start = int(lb[1]) + len(question) + 2
                    ans_end = int(lb[2]) + len(question) + 2
                    entity = lb[0]

                    # print(lb[3])                                  # In ra để kiểm tra xem khớp với câu trả lời có trong data chưa
                    # print(" ".join(se[ans_start:ans_end+1]))

                    label_idxs.append([entity, ans_start, ans_end])
                sample['label_idx'] = label_idxs
                l_sample.append(sample)
            else:                                                   # Nếu độ dài context lớn hơn max_seq
                idx = 0
                labels = self.list_sample[i]['label']
                while(len(context) > max_seq):                      # Cắt câu sử dụng sliding window cho đến khi len(context) < max_sep
                    label_idxs = []
                    ctx = context[:max_seq]
                    sent = question + ctx
                    context = context[self.stride:]
                    sample['context'] = ctx

                    char_seq = []
                    for word in sent:
                        character = self.get_character(word, self.max_char_len)
                        char_seq.append(character)
                    sample['char_sequence'] = char_seq

                    se = ['cls'] + question + ['sep'] +  ctx
                    
                    for j, lb in enumerate(labels):
                        start = int(lb[1]) + len(question) + 2
                        end = int(lb[2]) + len(question) + 2
                        
                        if start < (idx + 2 + len(question)) or end > (idx + len(sent)):            # Nếu start hoặc end không thuộc context được cắt thì start = 0, end = 0
                            start = 0
                            end = 0
                        else:
                            if start >= (idx + 2 + len(question)) and start <= (idx + len(sent)):   # Nếu start trong khoảng được cắt
                                start = start - idx
                            if end >= (idx + 2 + len(question)) and end <= (idx + len(sent)):       # Nếu end trong khoảng được cắt
                                end = end - idx
                        label_idxs.append([lb[0], start, end])

                    idx = idx + self.stride 

                    sample['label_idx'] = label_idxs
                    l_sample.append(sample)

        return l_sample


class MyDataSet(Dataset):

    def __init__(self, path, char_vocab_path, label_set_path,
                 max_char_len, tokenizer, max_seq_length, stride):

        self.samples = InputSample(path=path, max_char_len=max_char_len, max_seq_length=max_seq_length, stride=stride).get_sample()
        self.tokenizer = tokenizer                      # Chuyển văn bản thô thành số sử dụng tokenizer Phobert
                                                        # Example: "Chúng_tôi là những nghiên_cứu_viên ." -> tensor([[    0,   746,     8,    21, 46349,     5,     2]])             
        self.max_seq_length = max_seq_length            # Độ dài tối đa của context         
        self.max_char_len = max_char_len                # Độ dài tối đã của char
        with open(label_set_path, 'r', encoding='utf8') as f:       # Đọc file từ điển nhãn
            self.label_set = f.read().splitlines()

        with open(char_vocab_path, 'r', encoding='utf-8') as f:     # Đọc file từ điển các ký tự
            self.char_vocab = json.load(f)
        self.label_2int = {w: i for i, w in enumerate(self.label_set)}


    def preprocess(self, tokenizer, context, question, max_seq_length, mask_padding_with_zero=True):
        input_ids = [tokenizer.cls_token_id]                    # Thêm [CLS] vào đầu câu
        firstSWindices = [len(input_ids)]

        for w in question:
            word_token = tokenizer.encode(w)                    # Chuyển các token thành số
            input_ids += word_token[1: (len(word_token) - 1)]   # Chỉ lấy token đầu tiên
                                                                # Example: seq = "Chúng tôi"
                                                                # tokenizer.encode("Chúng tôi") -> [0, 746, 2]
                                                                # Lấy token đầu tiên tại vị trí [1: (len(word_token) - 1)]
            firstSWindices.append(len(input_ids))               # lưu lại vị trí token đã lấy 
        
        input_ids.append(tokenizer.sep_token_id)                # Thêm [SEP] và giữa question và context
        firstSWindices.append(len(input_ids))

        for w in context:
            word_token = tokenizer.encode(w)
            input_ids += word_token[1: (len(word_token) - 1)]
            if len(input_ids) >= max_seq_length:                
              firstSWindices.append(0)
            else:
              firstSWindices.append(len(input_ids))

        input_ids.append(tokenizer.sep_token_id)
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        if len(input_ids) > max_seq_length:             
            input_ids = input_ids[:max_seq_length]
            firstSWindices = firstSWindices + [0] * (max_seq_length - len(firstSWindices))
            firstSWindices = firstSWindices[:max_seq_length]
            attention_mask = attention_mask[:max_seq_length]
        else:
            attention_mask = attention_mask + [0 if mask_padding_with_zero else 1] * (max_seq_length - len(input_ids))
            input_ids = (
                    input_ids
                    + [
                        tokenizer.pad_token_id,
                    ]
                    * (max_seq_length - len(input_ids))
            )

            firstSWindices = firstSWindices + [0] * (max_seq_length - len(firstSWindices))

        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(firstSWindices)

    def character2id(self, character_sentence, max_seq_length):
        char_ids = []
        for word in character_sentence:                 # Lặp qua từng câu trong sentence
            word_char_ids = []
            for char in word:                           # Lặp qua từng ký tự trong câu
                if char not in self.char_vocab:         # Nếu ký tự không xuất hiện trong từ điển ký tự thì gán nó bằng 'UNK'
                    word_char_ids.append(self.char_vocab['UNK'])
                else:                                   # Nếu xuất hiện trong từ điển thì chuyển ký tự thành số 
                    word_char_ids.append(self.char_vocab[char])
            char_ids.append(word_char_ids)
        if len(char_ids) < max_seq_length:              # Nếu độ dài câu nhỏ hơn max_seq_length thì thêm padding vào cuối
            char_ids += [[self.char_vocab['PAD']] * self.max_char_len] * (max_seq_length - len(char_ids))
        return torch.tensor(char_ids)

    def span_maxtrix_label(self, label):        
        start, end, entity = [], [], []
        label = np.unique(label, axis=0).tolist()           # Loại bỏ những label trùng nhau
        for lb in label:                                    # lặp qua từng label
            if int(lb[1]) > self.max_seq_length or int(lb[2]) > self.max_seq_length:        # Nếu vị trí bd hoặc kết thúc lớn hơn max_seq_length thì chuyển thành vị trí (0, 0)
                start.append(0)
                end.append(0)
            else:
                start.append(int(lb[1]))
                end.append(int(lb[2]))
            try:
                entity.append(self.label_2int[lb[0]])
            except:
                print(lb)
        
        label = torch.sparse.FloatTensor(torch.tensor([start, end], dtype=torch.int64), torch.tensor(entity),
                                         torch.Size([self.max_seq_length, self.max_seq_length])).to_dense()
        # Example: start = 2, end = 3, entity = 1, max_seq_length = 5
        """ label = tensor([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
            label.shape = [max_seq_length, max_seq_length]"""
        return label

    def __getitem__(self, index):

        sample = self.samples[index]
        context = sample['context']
        question = sample['question']
        char_seq = sample['char_sequence']
        seq_length = len(question) + len(context) + 2        
        label = sample['label_idx']
        input_ids, attention_mask, firstSWindices = self.preprocess(self.tokenizer, context, question, self.max_seq_length)

        char_ids = self.character2id(char_seq, max_seq_length=self.max_seq_length)
        if seq_length > self.max_seq_length:
          seq_length = self.max_seq_length
        label = self.span_maxtrix_label(label)

        return input_ids, attention_mask, firstSWindices, torch.tensor([seq_length]), char_ids, label.long()

    def __len__(self):
        return len(self.samples)


def get_mask(max_length, seq_length):
    mask = [[1] * seq_length[i] + [0] * (max_length - seq_length[i]) for i in range(len(seq_length))]
    mask = torch.tensor(mask)
    mask = mask.unsqueeze(1).expand(-1, mask.shape[-1], -1)
    mask = torch.triu(mask)

    # Example: seq_length = [2], max_length = 5
    """ mask = tensor([[[1, 1, 1, 1, 0],        # start_seq = 0 , end_seq = seq_length
                        [0, 1, 1, 1, 0],        # start_seq = 1 , end_seq = seq_length
                        [0, 0, 1, 1, 0],        # start_seq = 2 , end_seq = seq_length
                        [0, 0, 0, 1, 0],        # start_seq = 3 , end_seq = seq_length
                        [0, 0, 0, 0, 0]]])      # start_seq = 4 , end_seq = seq_length """ 
    return mask

def get_useful_ones(out, label, mask):
    # get mask, mask the padding and down triangle

    mask = mask.reshape(-1)
    tmp_out = out.reshape(-1, out.shape[-1])
    tmp_label = label.reshape(-1)
    # index select, for gpu speed
    indices = mask.nonzero(as_tuple=False).squeeze(-1).long()
    tmp_out = tmp_out.index_select(0, indices)
    tmp_label = tmp_label.index_select(0, indices)

    return tmp_out, tmp_label