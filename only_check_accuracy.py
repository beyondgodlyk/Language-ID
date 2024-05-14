from transformers import AutoTokenizer, XLMRobertaModel
import torch
import torch.nn as nn

from datasets import load_dataset

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


class IdentificationModel(nn.Module):
    '''
    The main model for Language Identification
    Has a identifier which takes input based on conditions and outputs a language.
    '''
    def __init__(self, model, languages, use_mean_pooling, use_max_pooling):
        super(IdentificationModel, self).__init__()
        self.model = model
        self.total_num_pools = 1 + use_mean_pooling + use_max_pooling
        self.use_mean_pooling = use_mean_pooling
        self.use_max_pooling = use_max_pooling
        self.hidden_size = model.config.hidden_size
        self.languages = languages
        self.identifier = nn.Linear(self.total_num_pools * self.hidden_size, len(languages))

    def forward(self, src, attention_mask):
        outputs = self.model(src, attention_mask = attention_mask)
        identifier_input = outputs.pooler_output
        if self.use_mean_pooling:
            identifier_input = torch.cat([identifier_input, outputs.last_hidden_state.mean(dim=1)], dim=1)
        if self.use_max_pooling:
            identifier_input = torch.cat([identifier_input, outputs.last_hidden_state.max(dim=1).values], dim=1)
        return self.identifier(identifier_input)
    


class TokenizedDataset(Dataset):
    '''
    Only returns a tuple of tokenized texts and labels, which are later used in collate_fn for tokenization
    '''
    def __init__(self, raw_dataset, lang2id):
        self.texts = raw_dataset["text"]
        self.labels = raw_dataset["labels"]
        self.lang2id = lang2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.lang2id[self.labels[idx]]
        return text, label

def collate_fn(batch):
    '''
    Returns a tuple of lists for tokenized tokens, attention masks (used due to padding) and tensor labels respectively.
    '''
    texts = [tuple[0] for tuple in batch]
    labels = [tuple[1] for tuple in batch]

    tokenized_output = tokenizer(texts, return_tensors="pt", padding=True, max_length=32, truncation=True)
    tokenized_texts = tokenized_output["input_ids"]
    attention_masks = tokenized_output["attention_mask"]
    
    return tokenized_texts, attention_masks, torch.tensor(labels, dtype=torch.long)

if __name__ == "__main__":
    print("Starting to donwload model and dataset.")
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    model = XLMRobertaModel.from_pretrained("FacebookAI/xlm-roberta-base")

    dataset = load_dataset("papluca/language-identification")
    
    languages = sorted(dataset["train"].unique("labels"))
    lang2id = {lang: i for i, lang in enumerate(languages)}
    id2lang = {i: lang for lang, i in lang2id.items()}

    # Retrieve best model
    print("Loading model.")
    device = torch.device('cuda')
    best_model = IdentificationModel(model, languages, use_mean_pooling=True, use_max_pooling=True)

    best_model.load_state_dict(torch.load("best_model.pt", map_location=device))
    best_model.cuda()
    print(next(model.parameters()).is_cuda)
    print("Model loaded successfully.")
    test_dataset = TokenizedDataset(dataset["test"], lang2id)

    correct = 0
    total = test_dataset.__len__()
    print("Starting to check accuracy on test set.")
    for i, test_sample in enumerate(test_dataset):
        tokenized_output = tokenizer(test_sample[0], return_tensors="pt")
        tokenized_text = tokenized_output["input_ids"]
        attention_mask = tokenized_output["attention_mask"]
        label = torch.tensor(test_sample[1], dtype=torch.long)

        tokenized_text = tokenized_text.cuda()
        attention_mask.cuda()
        label.cuda()
        print(tokenized_text.get_device())
        print(attention_mask.get_device())
        output = best_model(tokenized_text, attention_mask)
        _, predicted = torch.max(output, 1)
        if predicted == label:
            correct += 1
        if i % 100 == 0:
            print(f"{i} samples done with accuracy {correct/(i+1)}.")
        # print(f"Predicted: {id2lang[predicted.item()]}, Actual: {id2lang[label.item()]}")
    print(f"Accuracy on test set is {correct/total}.")

    # Input a sentence
    inputs = tokenizer("What's happening", return_tensors="pt")
    outputs = best_model(inputs["input_ids"].cuda(), inputs["attention_mask"].cuda())
    print(id2lang[torch.argmax(outputs).item()])

