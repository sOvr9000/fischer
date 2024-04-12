
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer, Flatten, Dense
from fischer.keras_models.mdnn import MDNN



def main():
    # Define memory capacity of the model.
    mem_capacity = 2

    # Build a model to interpret its memory (a sequence of same-shaped N-D arrays).
    dnn = Sequential([
        InputLayer((mem_capacity, 3), name='dnn_input'), # Takes `mem_capacity` arrays of shape (3,).
        Flatten(), # Flatten the (2, 3) input to an array of shape (6,).
        Dense(8, 'relu'),
        Dense(3, 'softmax', name='dnn_output'), # Makes a prediction on the latest array of shape (3,) in its memory, which is automatically added to the end of the memory before prediction.
    ])

    # Construct the MDNN model around `dnn`.
    mdnn = MDNN(dnn)
    mdnn.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    mdnn.optimizer.learning_rate = 0.001
    mdnn.summary()

    # Generate simple data pairs (X, Y) where X = [a, b, c] and Y = n, where a, b, and c are random numbers sampled from a Gaussian distribution and n is the (wrapped) first difference of the argmax of the three random numbers.
    # For example:
    #   A first step with X = [0, 0, 1] implies the first difference is 0 (unchanged) and thus Y = 1.
    #   A second step with X = [0, 1, 0] implies the first difference is -1 and thus Y = 0.
    #   A third step with X = [0, 0, 1] implies the first difference is 1 and thus Y = 2.
    #   A fourth step with X = [1, 0, 0] implies the first difference is -2 and thus Y = 2.
    a = 0
    I3 = np.identity(3)
    for t in range(128):
        X = np.random.normal(0, 1, (1, 3))
        prev_a = a
        a = np.argmax(X[0])
        if t == 0:
            inc = 0
        else:
            inc = (a - prev_a + 1) % 3 - 1
        print(f'highest: {a}')
        print(f'increment: {inc}')

        # Sequential calls to this function builds the training data.
        mdnn.observe(X)

        # You must call this function before calling MDNN.observe() again.
        mdnn.label_previous_feature(I3[[inc + 1]])
    
    # Fit to the training data that is built from MDNN.observe().
    # Each step along the sequence is considered in training.
    # For example, a feature sequence [x_0, x_1, ..., x_N] with mapped labels [y_0, y_1, ..., y_N] implies
    # that the training will iterate over all feature sequences leading up to each label, such as the following:
    #   [x_0] -> y_0
    #   [x_0, x_1] -> y_1
    #   ...
    #   [x_0, x_1, ..., x_N] -> y_N
    mdnn.fit(epochs=8192)



if __name__ == '__main__':
    main()
