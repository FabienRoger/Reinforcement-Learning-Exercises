import pickle
import matplotlib.pyplot as plt

def plot(file):
    hist = pickle.load(open( file, 'rb' ))
    success = [1 if r > 0 else 0 for t,r in hist]
    x = 20
    accuracy_over_x_last = [sum(success[i:i+x])/x for i in range(len(success) - x)]
    plt.plot(accuracy_over_x_last)

plot('successful run 1.p')
plot('successful run 2.p')

plt.xlabel('run number')
plt.ylabel('last 20 accuracy (moving average)')
plt.show()