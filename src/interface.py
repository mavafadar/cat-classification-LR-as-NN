import math
from pathlib import Path

from model import *


def is_number(number, start, end, is_int):
    """
    This function checks if the given number is a valid number within a specified range.

    :param number: The number to be checked.
    :param start: The lower bound of the valid range.
    :param end: The upper bound of the valid range.
    :param is_int: A flag to indicate if the number should be an integer.
    :return bool: True if the number is a valid number within the specified range, and False otherwise.
    """
    if is_int:
        try:
            number = int(number)
        except ValueError:
            print("\nPlease enter a valid number.\n")
            return False
    if not is_int:
        try:
            number = float(number)
        except ValueError:
            print("\nPlease enter a valid number.\n")
            return False
    if not start <= number <= end:
        print(f"\nPlease enter a number in the range of [{start}, {end}].\n")
        return False
    return True


def train_model():
    """
    This function trains a model using a loaded dataset, and returns the trained model and other relevant information.

    The function loads the dataset, finds the number of pixels in the images, normalizes the dataset, and splits it into
    training and testing sets. The user is prompted to input the number of iterations for the gradient descent,
    the learning rate, and whether or not to plot the cost. The function then trains the model using the provided input.
    If the user chose to plot the cost, the cost is plotted after training.

    :return: A tuple containing the final trained model, the classes of the dataset, and the number of pixels in the images.
    """
    this_dataset = load_dataset()
    number_of_pixels = find_pixels_of_image(this_dataset)
    this_dataset = normalize_set_x(flatten_set_x(this_dataset))
    this_train_set_x, this_train_set_y = this_dataset["train_set_x"], this_dataset["train_set_y"]
    this_test_set_x, this_test_set_y = this_dataset["test_set_x"], this_dataset["test_set_y"]

    local_iterations_number = input("\nEnter the number of iterations for the gradient descent (Recommended: 2000): ")
    while not is_number(local_iterations_number, 1, math.inf, True):
        local_iterations_number = input(
            "\nEnter the number of iterations for the gradient descent (Recommended: 2000): ")
    local_learning_rate = input("Enter the learning rate for the gradient descent (Recommended: 0.005): ")
    while not is_number(local_learning_rate, 0, 1, False):
        local_learning_rate = input("Enter the learning rate for the gradient descent (Recommended: 0.005): ")
    is_plot_cost = input("Do you want the cost to be plotted after training (0: No, 1: Yes)? ")
    while not is_number(is_plot_cost, 0, 1, True):
        is_plot_cost = input("Do you want the cost to be plotted after training (0: No, 1: Yes)? ")

    local_final_model = model(this_train_set_x, this_train_set_y, this_test_set_x, this_test_set_y, int(local_iterations_number),
                              float(local_learning_rate), False)
    if int(is_plot_cost) == 1:
        plot_costs(local_final_model["costs"])

    return local_final_model, this_dataset["classes"], number_of_pixels


def check_image(this_model, this_classes, this_pixels):
    """
    This function allows the user to check an image using a pre-trained model, and prints the predicted class of the image.

    The function prompts the user to enter the name of the image, and verifies that the image exists in the images
    directory. The image is then passed to the test_your_image() function along with the pre-trained model, the number
    of pixels in the images, and the classes of the dataset.

    :param this_model: pre-trained model
    :param this_classes: classes of the dataset
    :param this_pixels: number of pixels in the images
    :return: None
    """
    image_name = input("\nPlease place your image in the images directory, and enter the name of the image: ")
    my_image = Path(f"./images/{image_name}")
    while not my_image.exists():
        image_name = input("\nThis image does not exist. Please enter a valid name: ")
        my_image = Path(f"./images/{image_name}")
    test_your_image(image_name, this_model, this_pixels, this_classes)


def startup_menu():
    """
    This function displays a menu of options for the user to select from, and prompts the user to input a valid choice.

    The function displays the options: train the model using the given dataset, test the model on your image, and exit.
    The user is prompted to input their choice. If the input is invalid, the user is prompted again.

    :return: The user's choice, either 1, 2, or 3.
    """
    print("1. Train the model using the given dataset.")
    print("2. Test the Model on Your Image.")
    print("3. Exit.\n")
    number = input("Select One of the Options: ")
    if not is_number(number, 1, 3, True):
        return startup_menu()
    return int(number)

