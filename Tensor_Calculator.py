import tensorflow as tf
import numpy as np

class TensorModule(tf.Module):
    def __init__(self, tensor):
        self.tensor = tf.Variable(tensor)

def tensor_calculator():
    print(r" _____                                ")
    print(r"|_   _|                               ")
    print(r"  | |    ___  _ __   ___   ___   _ __ ")
    print(r"  | |   / _ \| '_ \ / __| / _ \ | '__|")
    print(r"  | |  |  __/| | | |\__ \| (_) || |   ")
    print(r"  \_/   \___||_| |_||___/ \___/ |_|   ")
    print(r"                                      ")
    print("Welcome to the TensorFlow Tensor Calculator")

    def get_tensor_input(prompt):
        try:
            shape = tuple(map(int, input(f"Enter the shape of the {prompt} tensor (e.g., 2 3): ").split()))
            if len(shape) == 0:
                raise ValueError("Shape cannot be empty.")
            print(f"Enter the elements of the {prompt} tensor of shape {shape}:")
            elements = [float(input()) for _ in range(tf.reduce_prod(shape))]
            return tf.constant(elements, shape=shape)
        except ValueError as e:
            print(f"Invalid input: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    while True:
        print("\nAvailable operations:")
        print("1: Addition")
        print("2: Subtraction")
        print("3: Element-wise Multiplication")
        print("4: Element-wise Division")
        print("5: Matrix multiplication")
        print("6: Transpose")
        print("7: Reshape")
        print("8: Dot Product")
        print("9: Sum")
        print("10: Mean")
        print("11: Random Tensor Initialization")
        print("12: Standard Normal Distribution Tensor Initialization")
        print("13: Batch Reshape")
        print("14: Convolution")
        print("15: Activation Function (ReLU)")
        print("16: Max Pooling")
        print("17: Save Tensor to File")
        print("18: Load Tensor from File")
        print("19: Find Max Value")
        print("20: Find Min Value")
        print("21: Sigmoid Activation Function")
        print("22: Softmax Activation Function")
        print("23: Mean Squared Error")
        print("24: Train a Multilayer Perceptron (MLP)")
        print("0: Exit")

        operation = input("Choose an operation (0-24): ")

        if operation == '0':
            print("Exiting the calculator. Goodbye!")
            break

        try:
            if operation in ['1', '2', '3', '4', '5', '8']:
                tensor1 = get_tensor_input("first")
                tensor2 = get_tensor_input("second")
                if tensor1 is None or tensor2 is None:
                    print("Error in tensor input.")
                    continue

                if operation == '1':
                    result = tf.add(tensor1, tensor2)
                elif operation == '2':
                    result = tf.subtract(tensor1, tensor2)
                elif operation == '3':
                    result = tf.multiply(tensor1, tensor2)
                elif operation == '4':
                    result = tf.divide(tensor1, tensor2)
                elif operation == '5':
                    if tensor1.shape[-1] != tensor2.shape[0]:
                        print("Matrix multiplication not possible with these shapes.")
                        continue
                    result = tf.matmul(tensor1, tensor2)
                elif operation == '8':
                    result = tf.tensordot(tensor1, tensor2, axes=1)

            elif operation in ['6', '7', '9', '10']:
                tensor = get_tensor_input("single")
                if tensor is None:
                    print("Error in tensor input.")
                    continue
                if operation == '6':
                    result = tf.transpose(tensor)
                elif operation == '7':
                    new_shape = tuple(map(int, input("Enter the new shape (e.g., 3 2): ").split()))
                    result = tf.reshape(tensor, shape=new_shape)
                elif operation == '9':
                    result = tf.reduce_sum(tensor)
                elif operation == '10':
                    result = tf.reduce_mean(tensor)
            elif operation == '11':
                shape = tuple(map(int, input("Enter the shape of the tensor (e.g., 2 3): ").split()))
                result = tf.random.uniform(shape)
            elif operation == '12':
                shape = tuple(map(int, input("Enter the shape of the tensor (e.g., 2 3): ").split()))
                result = tf.random.normal(shape)
            elif operation == '13':
                tensor = get_tensor_input("single")
                if tensor is None:
                    print("Error in tensor input.")
                    continue
                batch_size = int(input("Enter the new batch size: "))
                result = tf.reshape(tensor, (batch_size, -1))
            elif operation == '14':
                tensor = get_tensor_input("input")
                kernel = get_tensor_input("kernel")
                if tensor is None or kernel is None:
                    print("Error in tensor input.")
                    continue
                tensor = tf.expand_dims(tensor, axis=0)
                kernel = tf.expand_dims(kernel, axis=-1)
                result = tf.nn.conv2d(tensor, kernel, strides=[1, 1, 1, 1], padding="VALID")
            elif operation == '15':
                tensor = get_tensor_input("single")
                if tensor is None:
                    print("Error in tensor input.")
                    continue
                result = tf.nn.relu(tensor)
            elif operation == '16':
                tensor = get_tensor_input("single")
                if tensor is None:
                    print("Error in tensor input.")
                    continue
                tensor = tf.expand_dims(tf.expand_dims(tensor, axis=0), axis=-1)
                pool_size = tuple(map(int, input("Enter the pool size (e.g., 2 2): ").split()))
                result = tf.nn.max_pool2d(tensor, ksize=pool_size, strides=pool_size, padding="VALID")
            elif operation == '17':
                tensor = get_tensor_input("single")
                if tensor is None:
                    print("Error in tensor input.")
                    continue
                filename = input("Enter the filename to save tensor: ")
                try:
                    module = TensorModule(tensor)
                    tf.saved_model.save(module, filename)
                    print(f"Tensor saved to {filename}")
                except Exception as e:
                    print(f"Error saving tensor: {e}")
                continue
            elif operation == '18':
                filename = input("Enter the filename to load tensor from: ")
                try:
                    module = tf.saved_model.load(filename)
                    result = module.tensor
                except Exception as e:
                    print(f"Error loading tensor: {e}")
                    continue
            elif operation == '19':
                tensor = get_tensor_input("single")
                if tensor is None:
                    print("Error in tensor input.")
                    continue
                result = tf.reduce_max(tensor)
            elif operation == '20':
                tensor = get_tensor_input("single")
                if tensor is None:
                    print("Error in tensor input.")
                    continue
                result = tf.reduce_min(tensor)
            elif operation == '21':
                tensor = get_tensor_input("single")
                if tensor is None:
                    print("Error in tensor input.")
                    continue
                result = tf.nn.sigmoid(tensor)
            elif operation == '22':
                tensor = get_tensor_input("single")
                if tensor is None:
                    print("Error in tensor input.")
                    continue
                result = tf.nn.softmax(tensor)
            elif operation == '23':
                tensor1 = get_tensor_input("first")
                tensor2 = get_tensor_input("second")
                if tensor1 is None or tensor2 is None:
                    print("Error in tensor input.")
                    continue
                result = tf.reduce_mean(tf.square(tensor1 - tensor2))
            elif operation == '24':
                try:
                    X = np.linspace(-1, 1, 100).reshape(-1, 1)
                    y = X**2 + np.random.normal(0, 0.05, (100, 1))

                    model = tf.keras.Sequential([
                        tf.keras.layers.Input(shape=(1,)),
                        tf.keras.layers.Dense(10, activation='relu'),
                        tf.keras.layers.Dense(10, activation='relu'),
                        tf.keras.layers.Dense(1)
                    ])
                    model.compile(optimizer='adam', loss='mse')

                    model.fit(X, y, epochs=50, verbose=0)

                    weights, biases = model.layers[0].get_weights()
                    print("Trained weights (first layer):", weights)
                    print("Trained biases (first layer):", biases)
                    continue
                except Exception as e:
                    print(f"Error training model: {e}")
                    continue
            else:
                print("Invalid operation.")
                continue

            print("Result:")
            print(result.numpy())
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    tensor_calculator()
