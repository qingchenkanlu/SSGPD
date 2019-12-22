
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot

# read the log file
fp = open('run.log', 'r')

train_epoch = []
train_loss = []
train_acc = []

test_epoch = []
test_loss = []
test_acc = []

for ln in fp:
  if 'Train' in ln:
    print ln
    print ln.split(',')
    print ln.split(',')[2][11:]
    train_epoch.append(int(ln.split(',')[0][6:]))
    train_loss.append(float(ln.split(',')[1][6:]))
    train_acc.append(float(ln.split(',')[2][11:]))

  if 'Test' in ln:
    print ln
    print ln.split(',')
    print ln.split(',')[2][11:]
    test_epoch.append(int(ln.split(',')[0][5:]))
    test_loss.append(float(ln.split(',')[1][6:]))
    test_acc.append(float(ln.split(',')[2][10:]))


fp.close()

""" plot train curve """
plt.figure(0)
host = host_subplot(111)
par1 = host.twinx()
# set labels
host.set_xlabel("epoch")
host.set_ylabel("train acc")
par1.set_ylabel("train loss")

# plot curves
p1, = host.plot(train_epoch, train_acc, label="train acc")
p2, = par1.plot(train_epoch, train_loss, label="train loss")

# set label color
host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())

plt.draw()


""" plot test curve """
plt.figure(1)
host = host_subplot(111)
par1 = host.twinx()
# set labels
host.set_xlabel("epoch")
host.set_ylabel("test acc")
par1.set_ylabel("test loss")

# plot curves
p1, = host.plot(test_epoch, test_acc, label="test acc")
p2, = par1.plot(test_epoch, test_loss, label="test loss")

# set label color
host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())

plt.draw()
plt.show()