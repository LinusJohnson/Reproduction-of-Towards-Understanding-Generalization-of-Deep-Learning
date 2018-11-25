import numpy as np
import os
import matplotlib.pyplot as plt
import re


def create_plot(folder):
    fig = plt.figure()
    for filename in os.listdir(folder):
        if filename != 'Figures':
            with open(os.path.join(os.getcwd(), folder, filename)) as f:
                w_time, step, accuracy = np.loadtxt(
                    f, delimiter=',', skiprows=1, unpack=True)
                mix_percent = re.search(
                    re.compile('(\d+)(?=_attack)'), filename).group(0)
                if mix_percent == '30':
                    mix_percent = '40'
                elif mix_percent == '40':
                    mix_percent = '60'
                plt.plot(step, accuracy, label=mix_percent + '.00%')
    plt.legend()
    fig_dir = os.path.join(folder, 'Figures')
    fig.savefig(
        os.path.join(fig_dir, 'accuracy_plot.png'),
        bbox_inches='tight',
        format='png',
        dpi=1200)


train_folder = os.path.join('csv_accuracy_data', 'Train')
test_folder = os.path.join('csv_accuracy_data', 'Test')
create_plot(train_folder)
create_plot(test_folder)