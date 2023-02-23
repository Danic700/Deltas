import matplotlib.pyplot as plt
import numpy as np





def make_plot(compression_time, compression_size, compression_algo, filename):
    plt.scatter(compression_time, compression_size)
    for i, label in enumerate(compression_algo):
        plt.annotate(label, (compression_time[i], compression_size[i]))
    # naming the x axis
    plt.xlabel('Compression Time (Seconds)')
    # naming the y axis
    plt.ylabel('Compression Size (Megabytes)')
    plt.xticks(np.arange(min(compression_time), max(compression_time), 50))
    plt.yticks(np.arange(min(compression_size), max(compression_size), 50))
    # giving a title to my graph
    plt.title(f"Cache Line Compression Results ({filename}) ")
    # function to show the plot
    plt.show()
    plt.savefig(f"{filename}-plot.png")


#