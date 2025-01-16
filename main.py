from A.cnn_classify import BreastMNIST_Classifier
if __name__ == '__main__':
    # task A
    breast_mnist_classifier = BreastMNIST_Classifier()
    # train model
    breast_mnist_classifier.train_model()
    # evaluate model
    breast_mnist_classifier.evaluate()
    # plot loss and accuracy
    breast_mnist_classifier.plot_curves()


    # task B
