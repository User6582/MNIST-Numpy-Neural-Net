import matplotlib.pyplot as plt
import pickle

scores  = []

for i in range(48,50):
    with open("save{0}.txt".format(i), "rb") as f:
        NN = pickle.load(f)
        scores.extend(NN[1][0])

plt.plot(scores)
plt.show()