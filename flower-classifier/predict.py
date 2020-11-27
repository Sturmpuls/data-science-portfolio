import argparse
import torch

from pathlib import Path

from classifier import ImageClassifier
from utils import ImageLoader
from utils import draw_random_image, labels_from_json, image_as_tensor


def main():
    parser = argparse.ArgumentParser(description='Predict most likely classes for an input image.')
    parser.add_argument('image_path',
                        type=str,
                        help='Image file')
    parser.add_argument('checkpoint',
                        type=str,
                        help='Checkpoint file')
    parser.add_argument('--top_k',
                        dest='top_k',
                        metavar='K',
                        type=int,
                        default=1,
                        help='Top K most likely classes')
    parser.add_argument('--category_names',
                        dest='labels',
                        metavar='C',
                        type=str,
                        default='cat_to_name.json',
                        help='Mapping of categories to labels.')
    parser.add_argument('--gpu',
                        dest='gpu',
                        type=bool,
                        nargs='?',
                        default=False,
                        const=True,
                        help='Use GPU for calculations.')
    parser.add_argument('--random',
                        dest='random',
                        type=bool,
                        nargs='?',
                        default=False,
                        const=True,
                        help='Predict random image from test dataset.')
    parser.add_argument('--data_dir',
                        dest='data_dir',
                        default='flowers',
                        type=str,
                        help='Only relevant for option --random')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.gpu else 'cpu'

    # load category mapping
    labels = labels_from_json(args.labels)

    # load the model
    model = ImageClassifier(device)
    model.load(args.checkpoint)
    
    # load random image
    if args.random:
        IL = ImageLoader(args.data_dir, labels)
        data = IL.datasets['test']
        args.image_path = draw_random_image(data)
        img_class = Path(args.image_path).parent.name
        img_label = IL.class_to_labels[str(img_class)]
        print(f'### Predicting for: {args.image_path} ({img_label}) ###')

    # predict
    predictions = model.predict(image_as_tensor(args.image_path), args.top_k, labels)

    for (prob, label) in predictions:
        print(f'{label}: {prob*100:.2f}%')


if __name__ == '__main__':
    main()