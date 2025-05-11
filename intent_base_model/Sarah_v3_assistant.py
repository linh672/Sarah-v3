import torch.nn as nn
import torch
import os
import json
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import random
import torch.nn as nn

class Sarah_v3_model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Sarah_v3_model, self).__init__()
        self.fc1 = nn.Linear(input_size, 128) #from 12888
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(0.5)
        

    def forward(self, X):
        X = self.relu(self.fc1(X))
        X = self.drop_out(X)
        X = self.relu(self.fc2(X))
        X = self.drop_out(X)
        X=  self.fc3(X)
        return X

class Sarah_v3_assistant:
    def __init__(self, intents_path, function_mappings = None):
        self.model = None
        self.intents_path = intents_path

        self.documents = [] #from json file
        self.vocabulary = [] #from response
        self.intents = [] #classifcation tags
        self.intents_responses = {} #save intents and related responses

        self.function_mappings = function_mappings #call funct if need

        self.X = None
        self.y = None
        
    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]
        return words
    
    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]
    
    def parse_intents(self):

        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)
            
            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']

                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag']))

                self.vocabulary = sorted(set(self.vocabulary))
    
    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)

        self.model = Sarah_v3_model(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]

        if self.function_mappings:
            if predicted_intent in self.function_mappings:
                self.function_mappings[predicted_intent]()

        if self.intents_responses[predicted_intent]:
            return random.choice(self.intents_responses[predicted_intent])
        else:
            return self.intents_responses.get(predicted_intent, ["Sarah don't understand, can you say again"])[0]
