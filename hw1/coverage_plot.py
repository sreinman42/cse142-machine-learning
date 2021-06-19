import matplotlib as mpl
import matplotlib.pyplot as plt

figure_width = 3
figure_height = 3

panel_width = 2 / figure_width
panel_height = 2 / figure_height

plt.figure(figsize=(10/3, 17/3))

labels = ["C1", "C2", "C3", "C4", "C5", "C6"]
x = [9, 5, 3, 2, 0, 0]
y = [16, 16, 13, 10, 4, 0]

N = 9
P = 16

panel1 = plt.axes([0.3, 0.1, 9/20, 16/20])
panel1.set_title("Coverage Plot")
panel1.set_xlim(-1, P+1)
panel1.set_ylim(-1, P+1)
panel1.set_xlabel("Negatives")
panel1.set_ylabel("Positives")
panel1.set_xticks([0, N])
panel1.set_xticklabels(['0', 'N'])
panel1.set_yticks([0, P])
panel1.set_yticklabels(['0', 'P'])

panel1.scatter(x, y)
for i, label in enumerate(labels):
    panel1.annotate(label, (x[i]+0.1, y[i]+0.1))

plt.savefig("coverage_plot.png", dpi=600)

plt.figure(figsize=(figure_width, figure_height))

panel2 = plt.axes([0.1675, 0.1675, panel_width, panel_height])
panel2.set_title("ROC Plot")
panel2.set_xlim(-0.1, 1.1)
panel2.set_ylim(-0.1, 1.1)
panel2.set_xlabel("Negatives")
panel2.set_ylabel("Positives")
panel2.set_xticks([0, 1])
panel2.set_yticks([0, 1])

x_n = [p / N for p in x]
y_n = [p / P for p in y]
panel2.scatter(x_n, y_n)
for i, label in enumerate(labels):
    panel2.annotate(label, (x_n[i]+0.01, y_n[i]+0.01))

plt.savefig("roc_plot.png", dpi=600)
