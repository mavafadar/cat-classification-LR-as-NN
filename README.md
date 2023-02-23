# Cat Classification using Logistic Regression as a Neural Network

This project is a simple implementation of a logistic regression as a neural network for cat classification. It also includes a Command-line User Interface (CLI) that allows users to train the model, check their own images, or exit the program.

## Installation
To use this project, you need to have `Python 3.x` installed on your machine. Then, clone this repository and install the required packages using the following command:

```
pip install -r requirements.txt
```

## Usage
To use the program, run the following command:

```
python ./src/main.py
```
This will start the CLI and prompt you to select an option.

**Option 1: Train the Model**

Selecting this option will prompt you to enter the number of iterations of gradient descent, the learning rate, and whether to plot the costs after training. After that, the program will train the model and print the train and test accuracy.

**Option 2: Check Your Image**

Selecting this option will prompt you to place your own images in the images folder and enter the name of the image you want to check. The program will then predict whether the image is a cat or not.

**Option 3: Exit**
Selecting this option will terminate the program.

## TODO
- [ ] Implement graphical user-interface.


## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

Feel free to modify and use the code as you see fit. If you have any questions or suggestions, please reach out to the author.

## Acknowledgments
- This project was inspired by Andrew Ng's Deep Learning course on Coursera.
- The dataset used in this project was provided by Coursera.
