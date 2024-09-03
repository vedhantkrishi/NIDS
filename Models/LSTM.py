from Imports import *
import config

class LSTMClassifier:
    def __init__(self, data):
        self.batch_size = config.batch_size
        self.loss_function = config.loss_function
        self.no_classes = config.no_classes
        self.no_epochs = config.no_epochs
        self.optimizer = config.optimizer
        self.verbosity = config.verbosity
        self.num_folds = config.num_folds
        self.dropout = config.dropout
        self.SEED = config.SEED
        self.acc_per_fold = []
        self.loss_per_fold = []
        self.skfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)
        self.fold_no = 1
        self.history = None
        self.X , self.Y = get_data(data)

    def fit(self):
        for train, test in self.skfold.split(self.X, self.Y):
            X_train = self.X.iloc[train].values
            x_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

            X_test = self.X.iloc[test].to_numpy()
            x_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

            y_train = np.array(self.Y.iloc[train])
            y_test = np.array(Y.iloc[test])

            lst = Sequential()
            lst.add(LSTM(50, return_sequences=True, input_shape=(1, X_train.shape[1])))
            lst.add(Dropout(self.dropout))
            lst.add(LSTM(50, return_sequences=True))
            lst.add(Dropout(self.dropout))
            lst.add(LSTM(50))
            lst.add(Dropout(self.dropout))

            lst.add(Dense(1, activation='sigmoid'))

            lst.compile(loss=self.loss_function,
                        optimizer=self.optimizer,
                        metrics=['accuracy'])
            print('------------------------------------------------------------------------')
            print(f'Training for fold {self.fold_no} ...')

            lst.summary()

            self.history = lst.fit(x_train, y_train, epochs=self.no_epochs, validation_split=0.2, verbose=self.verbosity)
            scores = lst.evaluate(x_test, y_test, verbose=0)
            print(f'Score for fold {self.fold_no}: {lst.metrics_names[0]} of {scores[0]}; {lst.metrics_names[1]} of {scores[1] * 100}%')
            self.acc_per_fold.append(scores[1] * 100)
            self.loss_per_fold.append(scores[0])

            self.fold_no = self.fold_no + 1

        print('Average scores for all folds:')
        print(f'> Accuracy: {np.mean(self.acc_per_fold)} (+- {np.std(self.acc_per_fold)})')
        print(f'> Loss: {np.mean(self.loss_per_fold)}')
        print('------------------------------------------------------------------------')

        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title("Plot of accuracy vs epoch for train and test dataset")
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='best')
        plt.savefig('Plot of accuracy vs epoch : LSTM')
        plt.show()

        # Plot of loss vs epoch of train and test dataset
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title("Plot of loss vs epoch for train and test dataset")
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='best')
        plt.savefig('Plot of loss vs epoch : LSTM')
        plt.show()

        y_classes = (lst.predict(x_test) > 0.5).astype('int32')

        print("Recall Score - ", recall_score(y_test, y_classes))
        print("F1 Score - ", f1_score(y_test, y_classes))
        print("Precision Score - ", precision_score(y_test, y_classes))



lstm_classifier = LSTMClassifier()
lstm_classifier.fit(X, Y)
