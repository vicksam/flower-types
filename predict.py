import argparse
import numpy as np
import utils

# Disable warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    args = parse_input_args()
    class_names, probabilities = predict(
        args.image_path,
        args.model_path,
        args.top_k,
        args.category_names
    )
    utils.print_results(class_names, probabilities)

# Create parser for parsing input args
# Return parsed args
def parse_input_args():
    parser = argparse.ArgumentParser(description='Flower image classifier')

    parser.add_argument(
        'image_path',
        action = 'store',
        type = str,
        help = 'input image path'
    )
    parser.add_argument(
        'model_path',
        action = 'store',
        type = str,
        help = 'classifier model path'
    )
    parser.add_argument(
        '--top_k',
        action = 'store',
        type = int,
        help = 'k value for top k most likely classes to be displayed',
        default = 5
    )
    parser.add_argument(
        '--category_names',
        action = 'store',
        type = str,
        help = 'path to json file with label-class dictionary'
    )

    return parser.parse_args()

def predict(image_path, model_path, top_k, class_names_path):
    image = utils.prepare_image(image_path)
    model = utils.load_model(model_path)

    # Predict
    predictions = model.predict(image)[0]

    # Get indexes of top k predictions
    # Sort indexes, in reverse (descending) order
    sorted_indexes = np.argsort(predictions)[::-1]
    top_k_indexes = sorted_indexes[0:top_k]

    # Extract probabilites and class names based on indexes
    probabilities = [predictions[i] for i in top_k_indexes]
    if class_names_path != None:
        # +1 because class_names key range is from 1 to 102
        class_names = utils.load_class_names(class_names_path, top_k_indexes)
    else:
        # Put indexes for class names
        class_names = [i for i in range(1, 103)]
    return class_names, probabilities

if __name__ == '__main__':
    main()
