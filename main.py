from A.cnn_classify import BreastMNIST_Classifier
from B.cnn_classify import BloodMNIST_Classifier
if __name__ == '__main__':
    # task A, Simple CNN
    for _ in range(5):
        breast_mnist_classifier = BreastMNIST_Classifier(model_name="SimpleCNN")
        # train and validate model
        breast_mnist_classifier.train_model()
        # test model
        breast_mnist_classifier.test()
        # plot loss and accuracy
        breast_mnist_classifier.plot_curves()


    # task B, Simple CNN
    for _ in range(5):
        blood_mnist_classifier = BloodMNIST_Classifier(model_name="SimpleCNN")
        # train and validate model
        blood_mnist_classifier.train_model()
        # evaluate model
        blood_mnist_classifier.test()
        # plot loss and accuracy
        blood_mnist_classifier.plot_curves()
