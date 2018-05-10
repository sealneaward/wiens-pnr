"""train_classifier.py

Usage:
    train_classifier.py <f_data_config> --binary
    train_classifier.py <f_data_config> --multi

Arguments:
    <f_data_config>  example ''data/config/wiens.yaml''

Example:
    python train_classifier.py wiens.yaml --binary
    python train_classifier.py wiens.yaml --multi
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# model
import tensorflow as tf
optimize_loss = tf.contrib.layers.optimize_loss

from wiens.data.dataset import BaseDataset
from wiens.data.extractor import BaseExtractor
from wiens.data.loader import GameSequenceLoader
import wiens.config as CONFIG

from sklearn.svm import SVC as SVM
from sklearn.metrics import classification_report, accuracy_score

from docopt import docopt
import yaml
import os

def train(data_config, label_type):
    # Initialize dataset/loader
    dataset = BaseDataset(data_config, 0, load_raw=False, no_anno=True)
    extractor = BaseExtractor(data_config)
    loader = GameSequenceLoader(
        dataset,
        extractor,
        data_config['batch_size']
    )

    if label_type == 'binary':
        val_x, val_t = loader.load_valid(decode=True)
        train_x, train_t = loader.load_train(decode=True)

        svm = SVM()
        svm.fit(train_x, train_t)
        predict = svm.predict(val_x)
        print(classification_report(val_t, predict))
        print('\n Accuracy: %s' % (accuracy_score(val_t, predict)))

    elif label_type == 'multi':
        val_x, val_z = loader.load_valid(decode=True, multi=True)
        train_x, train_z = loader.load_train(decode=True, multi=True)

        svm = SVM()
        svm.fit(train_x, train_z)
        predict = svm.predict(val_x)
        print(classification_report(val_z, predict))
        print('\n Accuracy: %s' % (accuracy_score(val_z, predict)))

if __name__ == '__main__':
    arguments = docopt(__doc__)
    print ("...Docopt... ")
    print(arguments)
    print ("............\n")
    f_data_config = '%s/%s' % (CONFIG.data.config.dir,arguments['<f_data_config>'])

    data_config = yaml.load(open(f_data_config, 'rb'))
    if arguments['--multi']:
        train(data_config, 'multi')
    elif arguments['--binary']:
        train(data_config, 'binary')
