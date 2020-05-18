from eunjeon import Mecab
from Models.RNNbased import convert_sentence, get_model, pred_sentences
from Models.RNNbased import best_model
import os

# set path
root_path = "C:/Users/user/PycharmProjects/NLP_SentimentAnalysis"
output_path = f"{root_path}/Output"
model_path = f"{output_path}/models"
stopwords_path = os.path.join(root_path + "/stopwords.txt")
tokenizer_path = os.path.join(root_path + "/tokenizer")

# tagger
mecab = Mecab()

# tokenizer : MANUALLY set, might change afterwards.
sent = convert_sentence(mecab, 80, stop_path=stopwords_path)
best_model = get_model(best_model(model_path, "*.h5"))
sent_pred = pred_sentences(sent, best_model)

