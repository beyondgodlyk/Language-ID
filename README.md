# Approach
I am using the pre-trained XLM-RoBERTa model and tokenizer and a classifier on top, which is used to identify the language of the query. I am using the pooled output, mean and max of hidden states as input to the classifier. Although using the mean and max of hidden states can be toggled using the flags of IdentificationModel class. 

I am using the [Language Identification dataset](https://huggingface.co/datasets/papluca/language-identification) which contains datasplit for train, validation and test sets. The training process is rather quick and the best model is found within 5 epochs. I am using the validation set for early stopping.

Finally this model achieves an accuracy of around 99.6 % (highest 99.62 %) which is same as the [available pre-trained model](https://huggingface.co/papluca/xlm-roberta-base-language-detection) on huggingface.com.

As the task mentioned, you can put your own input sentence in the last cell and use the model for identifying the language. Or if you want to execute using command line, the converted python script can be found on this [link](https://github.com/beyondgodlyk/Language-ID) in my github.
