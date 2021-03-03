from backinthessr.clean_data2 import get_data, clean_data
from backinthessr.model import train_model, evaluate
from backinthessr.clean_data2 import Vocabulary
import joblib
import dill

class Trainer():

    def train(self):
        # Get data
        X, y = get_data()
        X_clean = clean_data(X)

        # Compute X_pad and have voc
        voc = Vocabulary()
        voc.fit(X_clean)
        X_pad = voc.transform(X_clean)
        vocab_size = voc.vocab_size

        # X_train_pad, X_test_pad, y_train, y_test, vocab_size = clean_data(X, y)
        model = train_model(X_pad, y, vocab_size)
        # score = evaluate(model, X_test_pad, y_test)
        return model, voc
    
    def save(self, model, voc):
        model.save('model.h5')
        joblib.dump(voc, 'vocabulary.joblib')
        
        # with open("model.dill", "wb") as f:
            # dill.dump(model, f)
        pass

if __name__ == '__main__':
    trainer = Trainer()
    model, voc = trainer.train()
    trainer.save(model, voc)
