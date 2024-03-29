import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
import os
from sklearn.model_selection import train_test_split
import pandas as pd

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=7,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)



def getKobert():

    def predict(predict_sentence):

        emotion = ''

        data = [predict_sentence, '0']
        dataset_another = [data]

        another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
        test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)

        model1.eval()

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)

            valid_length= valid_length
            label = label.long().to(device)

            out = model1(token_ids, valid_length, segment_ids)


            test_eval=[]
            for i in out:
                logits=i
                logits = logits.detach().cpu().numpy()

                if np.argmax(logits) == 0:
                    test_eval.append("공포가")
                    emotion = "우울"
                elif np.argmax(logits) == 1:
                    test_eval.append("놀람이")
                    emotion = "우울"
                elif np.argmax(logits) == 2:
                    test_eval.append("분노가")
                    emotion = "우울"
                elif np.argmax(logits) == 3:
                    test_eval.append("슬픔이")
                    emotion = "우울"
                elif np.argmax(logits) == 4:
                    test_eval.append("중립이")
                    emotion = "중립"
                elif np.argmax(logits) == 5:
                    test_eval.append("행복이")
                    emotion = "행복"
                elif np.argmax(logits) == 6:
                    test_eval.append("혐오가")
                    emotion = "우울"

            print(test_eval[0] + " 느껴집니다.")
            return emotion




    
    device = torch.device("cuda:0" if torch.cuda.is_available() else  "cpu")
    bertmodel, vocab = get_pytorch_kobert_model()
    chatbot_data = pd.read_excel('/home2/siso/Documents/Dkobert/한국어_단발성_대화_데이터셋.xlsx', engine = 'openpyxl')
    chatbot_data.loc[(chatbot_data['Emotion'] == "공포"), 'Emotion'] = 0  #공포 => 0
    chatbot_data.loc[(chatbot_data['Emotion'] == "놀람"), 'Emotion'] = 1  #놀람 => 1
    chatbot_data.loc[(chatbot_data['Emotion'] == "분노"), 'Emotion'] = 2  #분노 => 2
    chatbot_data.loc[(chatbot_data['Emotion'] == "슬픔"), 'Emotion'] = 3  #슬픔 => 3
    chatbot_data.loc[(chatbot_data['Emotion'] == "중립"), 'Emotion'] = 4  #중립 => 4
    chatbot_data.loc[(chatbot_data['Emotion'] == "행복"), 'Emotion'] = 5  #행복 => 5
    chatbot_data.loc[(chatbot_data['Emotion'] == "혐오"), 'Emotion'] = 6  #혐오 => 6

    data_list = []
    for q, label in zip(chatbot_data['Sentence'], chatbot_data['Emotion'])  :
        data = []
        data.append(q)
        data.append(str(label))

        data_list.append(data)

    #train & test 데이터로 나누기
    dataset_train, dataset_test = train_test_split(data_list, test_size=0.25, random_state=0)


    ## Setting parameters
    max_len = 64
    batch_size = 64
    warmup_ratio = 0.1
    num_epochs = 10
    max_grad_norm = 1
    log_interval = 200
    learning_rate =  5e-5



    #토큰화
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
    data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)
    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)




    model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    os.chdir('/home2/siso/Documents/Dkobert/models/')

    model1 = torch.load('7emotions_model.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
    model1.load_state_dict(torch.load('7emotions_model_state_dict.pt'))  # state_dict를 불러 온 후, 모델에 저장

    checkpoint = torch.load('7emotions_all.tar')   # dict 불러오기
    model1.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    emotion = predict(sentence)
    return emotion
    
if __name__ == '__main__':
    main()
