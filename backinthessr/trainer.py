Class Trainer():

    def train(self):
        X, y = get_data()
        X_train_pad, X_test_pad = clean_data(X, y)
        model = train_model(X_train_pad, X_test_pad)
        score = evaluate(model)
        save(model)

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()