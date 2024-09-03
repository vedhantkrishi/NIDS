from Imports import *
import config
import data_load

class MLPClassifier:
    def __init__(self, data):
        self.batch_size = config.batch_size
        self.loss_function = config.loss_function
        self.no_classes = config.no_classes
        self.no_epochs = config.no_epochs
        self.optimizer = config.optimizer
        self.verbosity = config.verbosity
        self.num_folds = config.num_folds
        self.SEED = SEED
        self.acc_per_fold_mlp = []
        self.loss_per_fold_mlp = []
        self.skfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)
        self.fold_no = 1
        self.history = None
        self.X , self.Y = get_data(data)
    
    def fit(self):
        for train, test in self.skfold.split(self.X, self.Y):
            x_train = self.X.iloc[train].values
            x_test = self.X.iloc[test].to_numpy()
            y_train = np.array(self.Y.iloc[train])
            y_test = np.array(self.Y.iloc[test])
            mlp = Sequential()
            mlp.add(Dense(units=50, input_dim=x_train.shape[1], activation='relu'))
            mlp.add(Dense(1, activation='sigmoid'))
            mlp.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])
            print('------------------------------------------------------------------------')
            print(f'Training for fold {self.fold_no} ...')
            mlp.summary()
            self.history = mlp.fit(x_train, y_train, epochs=self.no_epochs, validation_split=0.2, verbose=self.verbosity)
            scores = mlp.evaluate(x_test, y_test, verbose=0)
            print(f'Score for fold {self.fold_no}: {mlp.metrics_names[0]} of {scores[0]}; {mlp.metrics_names[1]} of {scores[1]*100}%')
            self.acc_per_fold_mlp.append(scores[1] * 100)
            self.loss_per_fold_mlp.append(scores[0])
            self.fold_no += 1

    def display_results(self):
        print('Average scores for all folds:')
        print(f'> Accuracy: {np.mean(self.acc_per_fold_mlp)} (+- {np.std(self.acc_per_fold_mlp)})')
        print(f'> Loss: {np.mean(self.loss_per_fold_mlp)}')
        print('------------------------------------------------------------------------')
        
    def plot_accuracy(self):
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title("Plot of accuracy vs epoch for train and test dataset")
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='best')
        plt.savefig('Plot of accuracy vs epoch : MLP')
        plt.show()

# Assuming X and Y are already defined

# Creating an instance of the MLPClassifier class
mlp_classifier = MLPClassifier()

# Fitting the model
mlp_classifier.fit(X, Y)

# Displaying the results
mlp_classifier.display_results()

# Plotting the accuracy
mlp_classifier.plot_accuracy()
