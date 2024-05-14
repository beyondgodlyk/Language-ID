# %%
from transformers import AutoTokenizer, XLMRobertaModel
import torch
import torch.nn as nn

# Pretrained model of XLM-Roberta-Base
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
model = XLMRobertaModel.from_pretrained("FacebookAI/xlm-roberta-base")


# %%
from datasets import load_dataset

# Loading available dataset 
dataset = load_dataset("papluca/language-identification")
assert sorted(dataset["train"].unique("labels")) == sorted(dataset["validation"].unique("labels")) == sorted(dataset["test"].unique("labels"))

languages = sorted(dataset["train"].unique("labels"))
lang2id = {lang: i for i, lang in enumerate(languages)}
id2lang = {i: lang for lang, i in lang2id.items()}

# %%
# smol_dataset = {}
# smol_dataset["train"] = dataset["train"][:1000]

# smol_dataset["validation"] = dataset["validation"][:100]

# %%
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
    

# %%
from torch.utils.data import Dataset

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

# %%
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

train_dataset = TokenizedDataset(dataset["train"], lang2id)
train_loader = DataLoader(train_dataset, batch_size = 32, sampler = RandomSampler(train_dataset), collate_fn=collate_fn)

val_dataset = TokenizedDataset(dataset["validation"], lang2id)
val_loader = DataLoader(val_dataset, batch_size=32, sampler = SequentialSampler(val_dataset), collate_fn = collate_fn)

# %%
# it = train_loader._get_iterator()
# batch = next(it)
# batch

# %%
def train(imodel, train_loader, val_loader, max_epochs):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(imodel.parameters(), lr=1e-5)

    imodel.cuda()
    
    score_tracker = []
    epoch_for_best_score = -1
    for epoch in range(max_epochs):
        imodel.train()

        for batch in train_loader:
            tokenized_texts, attention_masks, labels = [x.cuda() for x in batch]
            output = imodel(tokenized_texts, attention_masks)
            loss_train = loss(output, labels)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            print(f"Loss for epoch {epoch} is {loss_train.item()}.")

        print(f"Training done for epoch {epoch}. Now checking performance on validation set.")
        # Check and save performance for validation set
        imodel.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for batch in val_loader:
                tokenized_texts, attention_masks, labels = [x.cuda() for x in batch]
                output = imodel(tokenized_texts, attention_masks)
                _, predicted = torch.max(output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = correct / total
            score_tracker.append(accuracy)
            if epoch_for_best_score == -1 or accuracy > max(score_tracker):
                epoch_for_best_score = epoch
                torch.save(imodel.state_dict(), "best_model.pt")
            if epoch - epoch_for_best_score > 5:
                break
        torch.save(imodel.state_dict(), f"model_{epoch}.pt")
        print(f"Accuracy for epoch {epoch} is {accuracy}.")
    return score_tracker

        

# %%
im = IdentificationModel(model, languages, use_mean_pooling=True, use_max_pooling=True)

score_tracker = train(im, train_loader, val_loader, 20)

# %% [markdown]
# Inference

# %%
# Retrieve best model
device = torch.device('cpu')
best_model = IdentificationModel(model, languages, use_mean_pooling=True, use_max_pooling=True)
best_model.load_state_dict(torch.load("best_model.pt"))

# %%
test_dataset = TokenizedDataset(dataset["test"], lang2id)

correct = 0
total = test_dataset.__len__()

for i, test_sample in enumerate(test_dataset):
    tokenized_text, attention_mask, label = collate_fn([test_sample])
    output = best_model(tokenized_text, attention_mask)
    _, predicted = torch.max(output, 1)
    if predicted == label:
        correct += 1
    if i % 100 == 0:
        print(f"{i} samples done with accuracy {correct/(i+1)}.")
    # print(f"Predicted: {id2lang[predicted.item()]}, Actual: {id2lang[label.item()]}")
print(f"Accuracy on test set is {correct/total}.")

# %%
# Input a sentence
inputs = tokenizer("What's happening", return_tensors="pt")
outputs = best_model(inputs["input_ids"], inputs["attention_mask"])
print(id2lang[torch.argmax(outputs).item()])


