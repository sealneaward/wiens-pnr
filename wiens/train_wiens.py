"""train_wiens.py

Usage:
    train_wiens.py <f_data_config> --binary
    train_wiens.py <f_data_config> --multi

Arguments:
    <f_data_config>  example ''data/config/wiens.yaml''

Example:
    python train_wiens.py wiens.yaml --binary
    python train_wiens.py wiens.yaml --multi
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# model
import tensorflow as tf
optimize_loss = tf.contrib.layers.optimize_loss

from wiens.data.dataset import BaseDataset
from wiens.data.extractor import BaseExtractor
from wiens.data.loader import WiensSequenceLoader
import wiens.config as CONFIG

from sklearn.svm import SVC as SVM
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, log_loss

from docopt import docopt
import yaml
import cPickle as pkl
import os

def train(data_config, label_type):
    # Initialize dataset/loader
    dataset = BaseDataset(data_config, 0, load_raw=False, no_anno=True)
    extractor = BaseExtractor(data_config)
    loader = WiensSequenceLoader(
        dataset,
        extractor,
        data_config['batch_size']
    )

    if label_type == 'binary':
        val_x, val_t = loader.load_valid(decode=True)
        train_x, train_t = loader.load_train(decode=True)

        svm = SVM(probability=True)
        svm.fit(train_x, train_t)

        lr = LogisticRegression()
        lr.fit(train_x, train_t)

        nn = MLPClassifier()
        nn.fit(train_x, train_t)

        rf = RandomForestClassifier()
        rf.fit(train_x, train_t)

        nb = GaussianNB()
        nb.fit(train_x, train_t)

        gb = GradientBoostingClassifier()
        gb.fit(train_x, train_t)

        svm_predict = svm.predict(val_x)
        svm_probs = svm.predict_proba(val_x)

        lr_predict = lr.predict(val_x)
        lr_probs = lr.predict_proba(val_x)

        rf_predict = rf.predict(val_x)
        rf_probs = rf.predict_proba(val_x)

        nb_predict = nb.predict(val_x)
        nb_probs = nb.predict_proba(val_x)

        nn_predict = nn.predict(val_x)
        nn_probs = nn.predict_proba(val_x)

        gb_predict = gb.predict(val_x)
        gb_probs = gb.predict_proba(val_x)

        # print('SVM: %s\n' % classification_report(val_t, svm_predict))
        # print('LR: %s\n' % classification_report(val_t, lr_predict))
        # print('NB: %s\n' % classification_report(val_t, nb_predict))
        # print('RF: %s\n' % classification_report(val_t, rf_predict))
        # print('NN: %s\n' % classification_report(val_t, nn_predict))
        # print('GB: %s\n' % classification_report(val_t, gb_predict))


        print('\n SVM Accuracy: %s' % (accuracy_score(val_t, svm_predict)))
        print('\n SVM Log Loss: %s' % (log_loss(val_t, svm_probs)))

        print('\n LR Accuracy: %s' % (accuracy_score(val_t, lr_predict)))
        print('\n LR Log Loss: %s' % (log_loss(val_t, lr_probs)))

        print('\n NB Accuracy: %s' % (accuracy_score(val_t, nb_predict)))
        print('\n NB Log Loss: %s' % (log_loss(val_t, nb_probs)))

        print('\n RF Accuracy: %s' % (accuracy_score(val_t, rf_predict)))
        print('\n RF Log Loss: %s' % (log_loss(val_t, rf_probs)))

        print('\n NN Accuracy: %s' % (accuracy_score(val_t, nn_predict)))
        print('\n NN Log Loss: %s' % (log_loss(val_t, nn_probs)))

        print('\n GB Accuracy: %s' % (accuracy_score(val_t, gb_predict)))
        print('\n GB Log Loss: %s' % (log_loss(val_t, gb_probs)))

        model_dict = {
            'svm': svm,
            'lr': lr,
            'gb': gb,
            'nn': nn,
            'rf': rf,
            'nb': nb
        }

        pkl.dump(model_dict, open('%s/wiens-models.pkl' % (CONFIG.model.dir), 'w'))


    elif label_type == 'multi':
        val_x, val_z = loader.load_valid(decode=True, multi=True)
        train_x, train_z = loader.load_train(decode=True, multi=True)

        svm = SVM(probability=True)
        svm.fit(train_x, train_z)
        predict = svm.predict(val_x)
        probs = svm.predict_proba(val_x)
        print(classification_report(val_z, predict))
        print('\n Accuracy: %s' % (accuracy_score(val_z, predict)))
        print('\n Log Loss: %s' % (log_loss(val_z, probs)))

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
