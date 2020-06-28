import argparse

def main():
    # Create parser for parsing input args
    parser = argparse.ArgumentParser(description='Flower image classifier')

    parser.add_argument(
        'image_path',
        action = 'store',
        type = str,
        help='input image path'
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
        help = 'k value for top k most likely classes',
        default = 5
    )
    parser.add_argument(
        '--category_names',
        action = 'store',
        type = str,
        help='path to json file with label-class dictionary'
    )

    # TODO: implement action based on arguments
    print(parser.parse_args())

if __name__ == '__main__':
    main()
