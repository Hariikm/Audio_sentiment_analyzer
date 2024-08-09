import whisper
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm.notebook import tqdm

MODEL= f'cardiffnlp/twitter-roberta-base-sentiment-latest'
tokenizer= AutoTokenizer.from_pretrained(MODEL)
model= AutoModelForSequenceClassification.from_pretrained(MODEL)
# Load the model
model_ = whisper.load_model("base")

whisper_model = model_


def sentiment(file_path):
    def transcribe_audio_file(file_path, model = whisper_model):
        # Transcribe the audio file
        result = model.transcribe(file_path)
        return result["text"]


    text1 = transcribe_audio_file(file_path)

    def polarity_score(example):
        encoded_text= tokenizer(example, return_tensors='pt')
        output=model(**encoded_text)
        scores=output[0][0].detach().numpy()
        scores=softmax(scores)
        scores_dict={
            'negative': scores[0],
            'neutral': scores[1],
            'positive': scores[2]
        }
        return scores_dict

    scores = polarity_score(text1)

    max_key = max(scores, key=scores.get)


    return max_key


