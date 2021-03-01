from backinthessr.clean_data import get_data, clean_data
from backinthessr.model import train_model, evaluate
import joblib

class Trainer():

    def train(self):
        X, y = get_data()
        X_train_pad, X_test_pad, y_train, y_test, vocab_size = clean_data(X, y)
        model = train_model(X_train_pad, y_train, vocab_size)
        score = evaluate(model, X_test_pad, y_test)
    
    def save(model):
        joblib.dump(model, 'model.joblib')

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()