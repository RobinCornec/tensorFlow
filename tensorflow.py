import getopt
import pickle
import sys
import time
from pathlib import Path
from random import uniform

import matplotlib.pyplot as plt
import tensorflow as tf

file_name = "polynome.dump"
file_name2 = "affine.dump"
my_file = Path(file_name)
my_file2 = Path(file_name2)




def main(argv):
    opts, args = getopt.getopt(argv, "ha:p:c:", ["help", "affine=", "polynome=","checking="])
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("You ash help")
            print(" -h, -- help : help")
            print(" -a, --affine <n>: run a learning session with n sessions and print the model ")
            print(" -p, --polynome <n> : run a learning session with n sessions and print the model")
            print(" -c, --checking <affine> or <polynome> : print a graph with model and reality ")
        elif opt in ("-a", "--affine"):
            nbre_lesson = int(arg)
            start_time = time.time()
            start_time_struct = time.gmtime(start_time)
            print("Debut de la simulation : " + time.strftime("%H:%M:%S", start_time_struct))
            print("Generation de l'échantillon d'apprentissage")
            data = []
            taille_apprentissage = 100
            for i in range(0, taille_apprentissage):
                print("\rProgression : " + str(round(((i + 1) / taille_apprentissage * 100), 3)), end='')
                x = uniform(0, 10)
                y = 2 * x + 6
                c = [x, y]
                data.append(c)
            print("Fin de generation de l'échantillon d'apprentissage")
            print("Allocation de la memoire pour X et Y")
            x = tf.placeholder("float")
            y = tf.placeholder("float")

            print("Recherche de la mémoire")
            w = load_affine()
            print("Generation du modele")
            y_model = tf.multiply(x, w[0]) + w[1]

            print("Définition de l'erreur")
            error = tf.square(y - y_model)
            print("Définition de l'entrainement")
            train_op = tf.train.GradientDescentOptimizer(0.02).minimize(error)

            print("Initialisation des variables")
            model = tf.global_variables_initializer()
            print("initialisation terminée")
            with tf.Session() as session:
                print("Debut de la phase d'apprentissage")
                session.run(model)
                for i in range(nbre_lesson):
                    for c in data:
                        instant_time = time.time()
                        elapse_time = instant_time - start_time
                        elapse_time_struct = time.gmtime(elapse_time)
                        print("\rProgression : " + str(round(((i + 1) / nbre_lesson * 100), 3)) +
                              " , temps ecoule : " + time.strftime("%H:%M:%S", elapse_time_struct), end='')
                        session.run(train_op, feed_dict={x: c[0], y: c[1]})
                w_value = session.run(w)
                print("Apprentissage terminé")
                print("Sauvegarde de la memoire")
                save_affine(w_value)
                print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))
        elif opt in ("-p", "--polynome"):
            nbre_lesson = int(arg)
            start_time = time.time()
            start_time_struct = time.gmtime(start_time)
            print("Debut de la simulation : " + time.strftime("%H:%M:%S", start_time_struct))
            print("Generation de l'échantillon d'apprentissage")
            data = []
            taille_apprentissage = 10000
            for i in range(0, taille_apprentissage):
                print("\rProgression : " + str(round(((i + 1) / taille_apprentissage * 100), 3)), end='')
                x = uniform(0, 10)
                y = pow(x, 3) * 2 + 3 * pow(x, 2) + 4 * x + 5
                c = [x, y]
                data.append(c)
            print("Fin de generation de l'échantillon d'apprentissage")
            print("Allocation de la memoire pour X et Y")
            x = tf.placeholder("float32")
            y = tf.placeholder("float32")

            print("Recherche de la mémoire")
            w = load_polynome()

            print("Generation du modele")
            y_model = w[0]
            for pow_i in range(1, 4):
                y_model = tf.add(tf.multiply(tf.pow(x, pow_i), w[pow_i]), y_model)

            print("Définition de l'erreur")
            error = tf.square(y - y_model) / 1000

            print("Définition de l'entrainement")
            train_op = tf.train.GradientDescentOptimizer(0.001).minimize(error)

            print("Initialisation des variables")
            model = tf.global_variables_initializer()
            print("initialisation terminée")
            with tf.Session() as session:
                print("Debut de la phase d'apprentissage")
                session.run(model)
                for i in range(nbre_lesson):
                    instant_time = time.time()
                    elapse_time = instant_time - start_time
                    elapse_time_struct = time.gmtime(elapse_time)
                    print("\rProgression : " + str(round(((i + 1) / nbre_lesson * 100), 3)) +
                          " , temps ecoule : " + time.strftime("%H:%M:%S", elapse_time_struct), end='')
                    for c in data:
                        session.run(train_op, feed_dict={x: c[0], y: c[1]})

                w_value = session.run(w)
                print("Apprentissage terminé")
                print("Sauvegarde de la memoire")
                save_polynome(w_value)

                print("Predicted model: {a:.3f}x^3 + {b:.3f}x^2+ {c:.3f}x + {d:.3f}".
                      format(a=w_value[3], b=w_value[2], c=w_value[1], d=w_value[0]))
        elif opt in ("-c", "--checking"):
            nom_function = str(arg)
            if nom_function == "affine":
                generate_graph_affine()
            elif nom_function == "polynome":
                generate_graph_polynome()
            else:
                print("WARNING : an argument is expected. Use --help or -h to more information")
        else:
            print("WARNING : a paramater is expected. Use --help or -h to more information")


if __name__ == '__main__':
    main(sys.argv[1:])

# Génère graph
def generate_graph_affine(a=2, b=6, tps=50):

    w = load_affine()
    x_values = []
    y = []
    res = []
    model = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(model)
        w_value = session.run(w)

    for i in range(0, tps, 2):
        x_values.append(i)
        res.append(w_value[0] * i + w_value[1])
        y.append((a * i) + b)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x_values, y, 'r', label="function")
    plt.plot(x_values, res, 'bo', label="neuron")
    # Now add the legend with some customizations.
    legend = plt.legend(loc='upper center', shadow=True)

    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width
    plt.show()


# Visualiser les valeurs réelles et celles calculées par le neurone.
def generate_graph_polynome(a=2, b=3, c=4, d=5, tps=25):
    w = load_polynome()
    x_values = []
    y = []
    res = []
    model = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(model)
        w_value = session.run(w)
    for i in range(-tps, tps, 2):
        x_values.append(i)
        res.append(w_value[3] * pow(i, 3) + w_value[2]*pow(i, 2)+w_value[1]*i+w_value[0])
        y.append((a * pow(i, 3) + b * pow(i, 2) + c * i + d))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x_values, y, 'r', label="function")
    plt.plot(x_values, res, 'bo', label="neuron")
    # Now add the legend with some customizations.
    legend = plt.legend(loc='upper center', shadow=True)

    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width
    plt.show()

# Savegarde
def save_polynome(w):

    print("i save my work")
    with open(file_name, "wb") as f:
        pickle.dump(w, f, protocol=2)


def save_affine(w):
    print("i save my work")
    with open(file_name2, "wb") as f:
        pickle.dump(w, f, protocol=2)

# Chargement
def load_polynome():
    if my_file.exists():
        print("Memory find")
        with open(file_name, "rb") as f:
            w = pickle.load(f)
            return tf.Variable(w)
    else:
        print("No memory to find")
        return tf.Variable([uniform(0, 0.1), uniform(0, 0.1), uniform(0, 0.1), uniform(0, 0.1)])


def load_affine():
    if my_file2.exists():
        print("Memory find")
        with open(file_name2, "rb") as f:
            w = pickle.load(f)
            return tf.Variable(w)
    else:
        print("No memory to find")
        return tf.Variable([uniform(0, 1), uniform(0, 1)])

