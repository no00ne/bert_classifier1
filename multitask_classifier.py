import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, test_model_multitask, model_eval_multitask

TQDM_DISABLE=True

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        ### TODO

        self.sentiment_classifier = nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)

        # 添加用于释义检测的层
        self.paraphrase_classifier = nn.Linear(BERT_HIDDEN_SIZE, 1)

        # 添加用于语义文本相似性的层
        self.similarity_regressor = nn.Linear(BERT_HIDDEN_SIZE, 1)


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        outputs = self.bert(input_ids, attention_mask)
        cls_output = outputs['pooler_output']

        return cls_output


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        cls_output = self.forward(input_ids, attention_mask)
        logits = self.sentiment_classifier(cls_output)
        return logits


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        cls_output_1 = self.forward(input_ids_1, attention_mask_1)
        cls_output_2 = self.forward(input_ids_2, attention_mask_2)
        diff = torch.abs(cls_output_1 - cls_output_2)
        logits = self.paraphrase_classifier(diff)
        return logits


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        cls_output_1 = self.forward(input_ids_1, attention_mask_1)
        cls_output_2 = self.forward(input_ids_2, attention_mask_2)
        diff = torch.abs(cls_output_1 - cls_output_2)
        logits = self.similarity_regressor(diff)
        return logits




def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


## Currently only trains on sst dataset
def load_model(filepath):
    checkpoint = torch.load(filepath)
    config = checkpoint['model_config']
    model = MultitaskBERT(config)
    model.load_state_dict(checkpoint['model'])
    optimizer = AdamW(model.parameters())
    optimizer.load_state_dict(checkpoint['optim'])

    random.setstate(checkpoint['system_rng'])
    np.random.set_state(checkpoint['numpy_rng'])
    torch.random.set_rng_state(checkpoint['torch_rng'])

    return model, optimizer, checkpoint['args']


def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sst_dev_data.collate_fn)
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=para_dev_data.collate_fn)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sts_dev_data.collate_fn)


    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    if os.path.exists(args.filepath_of_pretrain):
        model, optimizer, loaded_args = load_model(args.filepath_of_pretrain)
        print(f"Loaded pretrained model from {args.filepath_of_pretrain}")
    else:
        model = MultitaskBERT(config)
        optimizer = AdamW(model.parameters(), lr=args.lr)
        print(f"No pretrained model found. Initializing new model.")

    model = model.to(device)
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    if args.option == 'finetune':
        for param in model.bert.parameters():
            param.requires_grad = True

    best_dev_acc = 0

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        total_loss_sst = 0
        total_loss_para = 0
        total_loss_sts = 0

        for sst_batch, para_batch, sts_batch in zip(tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE),
                                                    para_train_dataloader, sts_train_dataloader):
            # Sentiment Classification Task
            sst_ids, sst_mask, sst_labels = (sst_batch['token_ids'],
                                             sst_batch['attention_mask'], sst_batch['labels'])
            sst_ids = sst_ids.to(device)
            sst_mask = sst_mask.to(device)
            sst_labels = sst_labels.to(device)
            logits_sst = model.predict_sentiment(sst_ids, sst_mask)
            loss_sst = F.cross_entropy(logits_sst, sst_labels.view(-1), reduction='sum') / args.batch_size

            # Paraphrase Detection Task
            para_ids_1, para_mask_1, para_ids_2, para_mask_2, para_labels = (para_batch['token_ids_1'],
                                                                             para_batch['attention_mask_1'],
                                                                             para_batch['token_ids_2'],
                                                                             para_batch['attention_mask_2'],
                                                                             para_batch['labels'])
            para_ids_1 = para_ids_1.to(device)
            para_mask_1 = para_mask_1.to(device)
            para_ids_2 = para_ids_2.to(device)
            para_mask_2 = para_mask_2.to(device)
            para_labels = para_labels.to(device)
            logits_para = model.predict_paraphrase(para_ids_1, para_mask_1, para_ids_2, para_mask_2).squeeze()
            loss_para = F.binary_cross_entropy_with_logits(logits_para, para_labels.float(), reduction='sum') / args.batch_size

            # Semantic Textual Similarity Task
            sts_ids_1, sts_mask_1, sts_ids_2, sts_mask_2, sts_labels = (sts_batch['token_ids_1'],
                                                                        sts_batch['attention_mask_1'],
                                                                        sts_batch['token_ids_2'],
                                                                        sts_batch['attention_mask_2'],
                                                                        sts_batch['labels'])
            sts_ids_1 = sts_ids_1.to(device)
            sts_mask_1 = sts_mask_1.to(device)
            sts_ids_2 = sts_ids_2.to(device)
            sts_mask_2 = sts_mask_2.to(device)
            sts_labels = sts_labels.to(device).float()  # Convert to Float here
            logits_sts = model.predict_similarity(sts_ids_1, sts_mask_1, sts_ids_2, sts_mask_2).squeeze()
            loss_sts = F.mse_loss(logits_sts, sts_labels.view(-1), reduction='sum') / args.batch_size

            # Combine the losses
            total_loss = loss_sst + loss_para + loss_sts

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            total_loss_sst += loss_sst.item()
            total_loss_para += loss_para.item()
            total_loss_sts += loss_sts.item()
            num_batches += 1

        train_loss = train_loss / num_batches
        total_loss_sst = total_loss_sst / num_batches
        total_loss_para = total_loss_para / num_batches
        total_loss_sts = total_loss_sts / num_batches

        train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
        # Evaluate on dev set
        paraphrase_accuracy, _, _, sentiment_accuracy, _, _, sts_corr, _, _ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)

        if sentiment_accuracy > best_dev_acc:
            best_dev_acc = sentiment_accuracy
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, "
              f"train acc :: {train_acc :.3f}, dev acc :: {sentiment_accuracy :.3f}")
        print(f"loss_sst :: {total_loss_sst :.3f}, loss_para :: {total_loss_para :.3f}, loss_sts :: {total_loss_sts :.3f}")
        print(f"Paraphrase detection accuracy: {paraphrase_accuracy:.3f}")
        print(f"Semantic Textual Similarity correlation: {sts_corr:.3f}")



def test_model(args):
    os.makedirs(os.path.dirname(args.sst_dev_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.sst_test_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.para_dev_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.para_test_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.sts_dev_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.sts_test_out), exist_ok=True)
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="finetune")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    parser.add_argument("--filepath_of_pretrain", type=str, default="None")
    args = parser.parse_args()
    return args

if __name__ == "__main__":


    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
