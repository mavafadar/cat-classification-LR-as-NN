from interface import *


def main():
    """
    The function provides an interactive menu-driven interface for the user to train a model using a dataset, test the
    model on an image, or exit the program.

    The function first welcomes the user and initializes the variables to store the final trained model, classes
    and pixels. It enters a loop to display the options menu and prompts the user to select an option.
    If the user chooses to train the model, the model is trained using the train_model() function, and the final model,
    classes, and pixels are stored. If the user chooses to test the model on an image, the check_image() function is
    called with the stored final model, classes, and pixels. If the user chooses to exit, a message is displayed and
    the program exits.

    :return: None
    """
    print("Welcome to the image classifier using logistic regression.\n")
    final_model, classes, pixels = None, None, None
    while True:
        option = startup_menu()
        if option == 1:
            final_model, classes, pixels = train_model()
        if option == 2:
            if final_model is None:
                print("\nYou have to train the model first.\n")
            else:
                check_image(final_model, classes, pixels)
        if option == 3:
            print("\nThank you for using this application.")
            break


if __name__ == "__main__":
    main()

