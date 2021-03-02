from backinthessr.clean_data import get_data, clean_data
from backinthessr.model import train_model, evaluate
import joblib
import dill

class Trainer():

    def train(self):
        X, y = get_data()
        X_clean = 
        X_pad = 
        vocab_size = 

        # X_train_pad, X_test_pad, y_train, y_test, vocab_size = clean_data(X, y)
        model = train_model(X_pad, y, vocab_size)
        # score = evaluate(model, X_test_pad, y_test)
        return model
    
    def save(self, model):
        model.save('model.h5')
        # joblib.dump(model, 'model.joblib')
        # with open("model.dill", "wb") as f:
            # dill.dump(model, f)
        pass

if __name__ == '__main__':
    trainer = Trainer()
    model = trainer.train()
    trainer.save(model)
