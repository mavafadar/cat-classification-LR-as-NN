import h5py
import imageio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_dataset():
    """
    This function loads the "Cat vs Non-Cat" dataset from two .h5 files: "TrainCatVsNonCat.h5" and "TestCatVsNonCat.h5".

    :return: A dictionary with the following keys:
    - "train_set_x": A shallow copy of the train set features (np.array).
    - "train_set_y": A shallow copy of the train set labels (np.array).
    - "test_set_x": A shallow copy of the test set features (np.array).
    - "test_set_y": A shallow copy of the test set labels (np.array).
    - "classes": A shallow copy of the list of classes (np.array).
    """
    train_dataset = h5py.File("./datasets/TrainCatVsNonCat.h5")
    # A shallow copy of the train set features:
    train_set_x = np.array(train_dataset["train_set_x"][:])
    # A shallow copy of the train set labels:
    train_set_y = np.array(train_dataset["train_set_y"][:])
    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))

    test_dataset = h5py.File("./datasets/TestCatVsNonCat.h5")
    # A shallow copy of the test set features:
    test_set_x = np.array(test_dataset["test_set_x"][:])
    # A shallow copy of the test set labels:
    test_set_y = np.array(test_dataset["test_set_y"][:])
    test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))

    # A shallow copy of the list of classes:
    classes = np.array(test_dataset["list_classes"][:])

    dataset = {
        "train_set_x": train_set_x,
        "train_set_y": train_set_y,
        "test_set_x": test_set_x,
        "test_set_y": test_set_y,
        "classes": classes,
    }

    return dataset


def show_image(index, dataset):
    """
    This function is used to display an image from a dataset, along with its corresponding label.

    :param index: The index of the image in the dataset that you wish to display.
    :param dataset: A dictionary containing the following keys: "train_set_x", "train_set_y", "classes".
    :return: None
    """
    train_set_x, train_set_y, classes = dataset["train_set_x"], dataset["train_set_y"], dataset["classes"]
    plt.imshow(train_set_x[index])
    print(f"y = {train_set_y[0, index]}, it is a {classes[np.squeeze(train_set_y[:, index])].decode('utf-8')} picture.")
    plt.show()


def find_number_of_examples(dataset):
    """
    This function takes in a dataset as an input and returns a dictionary containing the number of examples in the
    training set and the test set.

    :param dataset: A dictionary containing the keys 'train_set_x' and 'test_set_x' with the respective training and
    test sets.
    :return: A dictionary containing the keys 'm_train' and 'm_test' representing the number of examples in the
    training and test sets respectively.
    """
    train_set_x, test_set_x = dataset["train_set_x"], dataset["test_set_x"]
    m_train = train_set_x.shape[0]
    m_test = test_set_x.shape[0]

    number_of_examples = {
        "m_train": m_train,
        "m_test": m_test,
    }

    return number_of_examples


def find_pixels_of_image(dataset):
    """
    Given a dataset, finds the number of pixels in the image.

    :param dataset: Dictionary containing the dataset.
    :return: Number of pixels in the image.
    """
    train_set_x = dataset["train_set_x"]
    number_of_pixels = train_set_x.shape[1]
    return number_of_pixels


def flatten_set_x(dataset):
    """
    Flatten the train and test set x data in the dataset and return the updated dataset.

    :param dataset: A dictionary containing the train and test set x data.
    :return: The updated dataset with flattened train and test set x data.
    """
    train_set_x, test_set_x = dataset["train_set_x"], dataset["test_set_x"]
    dataset["train_set_x"] = train_set_x.reshape(train_set_x.shape[0], -1).T
    dataset["test_set_x"] = test_set_x.reshape(test_set_x.shape[0], -1).T
    return dataset


def normalize_set_x(dataset):
    """
    Normalize the train and test set of x values in the given dataset by dividing each value by 255.

    :param dataset: Dictionary containing the train and test set of x values.
    :return: Normalized dataset with train and test set of x values.
    """
    train_set_x, test_set_x = dataset["train_set_x"], dataset["test_set_x"]
    dataset["train_set_x"] = train_set_x / 255
    dataset["test_set_x"] = test_set_x / 255
    return dataset


def sigmoid(z):
    """
    The function sigmoid(z) returns the sigmoid of the input value z. The sigmoid function is defined as
    1 / (1 + e^(-z)), where e is the base of the natural logarithm and z is the input value.
    The sigmoid function is often used in machine learning as an activation function for artificial neural networks.
    It maps any input value to a value between 0 and 1, making it useful for binary classification tasks.

    :param z: A float or numpy array of any size which is the value to the sigmoid function.
    :return: A float or a numpy array which is the sigmoid of the input value.
    """
    return 1 / (1 + np.exp(-z))


def initialize_with_zeros(dimension):
    """
    his function initializes the weights and bias for a given dimension of the model with zeros.

    :param dimension: An integer which is the dimension of the model.
    :return: A dictionary containing the keys 'w' and 'b' with values of the weights and bias respectively.
             w is of shape (dimension, 1) and b is of type int and both initialized with zeros.
    """
    weights_and_biases = {
        "w": np.zeros((dimension, 1)),
        "b": 0,
    }

    return weights_and_biases


def propagate(w, b, X, Y):
    """
    Propagates the input forward through the network and calculates gradients and cost.

    :param w: An np.ndarray which is The weight matrix of the network.
    :param b: An np.ndarray which is The bias vector of the network.
    :param X: An np.ndarray which is The input data, of shape (n_x, m) where n_x is the number of features and m is
              the number of examples
    :param Y: An np.ndarray which is The true labels of the input data, of shape (1, m).
    :return: A tuple containing:
             gradients (dict): A dictionary containing the calculated gradients for w and b. Keys are "dw" and "db".
             cost (float): The calculated cost of the current input and parameters.
    """
    m = X.shape[1]
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)

    cost = np.sum(-Y * np.log(A) - (1 - Y) * np.log(1 - A)) / m

    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m

    gradients = {
        "dw": dw,
        "db": db,
    }

    return gradients, cost


def optimize(w, b, X, Y, iterations_number, learning_rate, print_cost=False):
    """
    Optimize the parameters of a linear model using gradient descent.

    :param w: A numpy array which is the weights of the linear model.
    :param b: A float which is the bias of the linear model.
    :param X: A numpy array which is the input data.
    :param Y: A numpy array which is the output data.
    :param iterations_number: An integer which is the number of iterations to perform gradient descent.
    :param learning_rate: A float which is the learning rate of the gradient descent algorithm.
    :param print_cost: An optional boolean. If set to True, the cost after each iteration will be printed. Default is False.
    :return: A tuple containing:
             parameters: A dictionary containing the optimized weights and bias.
             gradients: A dictionary containing the final gradients of the weights and bias
             costs : A list which is the list of costs at every 100th iteration.
    """
    costs = list()
    dw, db = 0, 0
    for iteration in range(iterations_number):
        gradients, cost = propagate(w, b, X, Y)
        dw, db = gradients["dw"], gradients["db"]
        w -= learning_rate * dw
        b -= learning_rate * db

        if iteration % 100 == 0:
            costs.append(cost)

        if print_cost and iteration % 100 == 0:
            print(f"Cost after iteration #{iteration}: {cost}")

    parameters = {
        "w": w,
        "b": b,
    }

    gradients = {
        "dw": dw,
        "db": db,
    }

    return parameters, gradients, costs


def plot_costs(costs):
    """
    Plots the costs in a line graph.

    :param costs: A list of costs to be plotted.
    :return: None.
    """
    plt.plot(costs)
    plt.ylabel("cost")
    plt.xlabel("iterations (per hundreds)")
    plt.show()


def predict(w, b, X):
    """
    Predict outputs of a binary classification model.

    This function takes in the weights (w), bias (b) and input data (X) of a binary classification model and
    returns the predicted outputs (Y_prediction). The prediction is based on the sigmoid activation function,
    where the output is 1 if the sigmoid function output is greater than or equal to 0.5, and 0 otherwise.

    :param w: A numpy array which is the weights of the model, reshaped to match the shape of the input data.
    :param b: A float which is the bias term of the model.
    :param X: A numpy array which is the input data for which predictions are to be made
    :return: A numpy array A binary array of predicted outputs, where 1 represents the positive class and 0
             represents the negative class.
    """
    w = w.reshape(X.shape[0], 1)
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    Y_prediction = (A >= 0.5) * 1.0
    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, iterations_number=2000, learning_rate=0.5, print_cost=False):
    """
    Trains a logistic regression model on given data and returns the final results.

    :param X_train: An np.array which is the input training data
    :param Y_train: An np.array which is the output training data
    :param X_test: An np.array which is the input testing data
    :param Y_test: An np.array which is the output testing data
    :param iterations_number: An optional integer which is the number of iterations to perform during optimization.
                              Defaults to 2000.
    :param learning_rate: An optional float which is the learning rate for gradient descent. Defaults to 0.5.
    :param print_cost: An optional boolean which determines whether to print the cost during optimization.
                       Defaults to False.
    :return: A dictionary containing the final results including:
             - costs (list): List of costs during optimization.
             - Y_prediction_test (np.array): Predicted output for the test data
             - Y_prediction_train (np.array): Predicted output for the train data
             - w (np.array): The final learned weights of the model
             - b (float): The final learned bias of the model
             - learning_rate (float): The learning rate used during optimization
             - iterations_number (int): The number of iterations used during optimization.
    """
    weights_and_biases = initialize_with_zeros(X_train.shape[0])
    w, b = weights_and_biases["w"], weights_and_biases["b"]
    parameters, gradients, costs = optimize(w, b, X_train, Y_train, iterations_number, learning_rate, print_cost)
    w, b = parameters["w"], parameters["b"]
    Y_prediction_train, Y_prediction_test = predict(w, b, X_train), predict(w, b, X_test)

    print(f"\nTrain Accuracy: {100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100}%")
    print(f"Test Accuracy: {100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100}%\n")

    final_model = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "iterations_number": iterations_number
    }

    return final_model


def test_your_image(image_name, final_result, pixels_number, classes):
    """
    This function takes in an image file, final model parameters, number of pixels, and a list of class labels
    and displays the image with the predicted class label.

    :param image_name: A string which is the name of the image file to be tested in the images directory.
    :param final_result: A dictionary containing the final model's weights and biases.
    :param pixels_number: An integer which is the number of pixels for the image.
                          The image dimensions should be changed to this number.
    :param classes: A list of class labels.
    :return: None.
    """
    file_name = f"./images/{image_name}"

    image = np.array(imageio.v3.imread(file_name))
    my_image = np.array(Image.fromarray(image).resize((pixels_number, pixels_number)))
    my_image = my_image.reshape((1, pixels_number * pixels_number * 3)).T
    my_image = my_image / 255.0
    my_predicted_image = np.squeeze(predict(final_result["w"], final_result["b"], my_image))
    print(f"y = {my_predicted_image} your algorithm predicts a \"{classes[int(my_predicted_image)].decode('utf-8')}\".\n")
    plt.imshow(image)
    plt.show()
