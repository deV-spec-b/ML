import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

gipers = pd.read_csv(r"Гиперспектр кукурузы.csv", sep=';', decimal=',')

Y = gipers['Spectr']
x = gipers['wavelength']

plt.figure()
plt.scatter(x, Y)
plt.grid(True)
plt.show()

#1
a1 = ((x - x.mean())*(Y - Y.mean())).mean() / ((x - x.mean())**2).mean()
a0 = Y.mean() - a1 * x.mean()

print("Модель линейной регрессии: Y^ =", a0, "+", a1, "* x")

#серия прогнозов
x_space = np.linspace(x.min(), x.max(), 100)
Y_pred = a0 + a1*x_space


fig  = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.scatter(x, Y)
ax.plot(x_space, Y_pred, 'r', label='Аналитическое решение')
ax.legend()
plt.grid(True)
plt.show()



#2 
class SimpleRegression(object):
    def __init__(self):
        self.a0 = 0
        self.a1 = 0
    def predict(self, x):
        return self.a0 + self.a1*x
    def MSE(self, x, Y):
        return ((Y - self.predict(x))**2).mean()
    def fit(self, x, Y, alpha=0.000000001, epsylon = 0.0001, max_steps = 5000):
        steps, errors = [], []
        step = 0
        for _ in range(max_steps):
            dT_a0 = -2*(Y - self.predict(x)).mean()
            dT_a1 = -2*((Y - self.predict(x))*x).mean()
            self.a0 -= alpha*dT_a0
            self.a1 -= alpha*dT_a1
            new_error = self.MSE(x, Y)
            step += 1
            steps.append(step)
            errors.append(new_error)
            if new_error < epsylon:
                break
        return steps, errors
    
    
regr = SimpleRegression()
steps, errors = regr.fit(x, Y, alpha=0.000000001, epsylon=0.0001, max_steps=5000)

plt.figure()
plt.plot(steps, errors)
plt.grid(True)
plt.show()

x_space = np.linspace(x.min(), x.max(), 100)
Y_pred_gs = regr.predict(x_space)

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.scatter(x, Y)
ax.plot(x_space, Y_pred_gs, 'y', label='Градиентный спуск')
ax.legend()
ax.grid(True)
plt.show()


#3 
print(f'Аналитическое решение: a0={a0}, a1={a1}, MSE={((Y - (a0 + a1 * x))**2).mean()}')
print(f'Градиентный спуск: a0={regr.a0}, a1={regr.a1}, MSE={regr.MSE(x, Y)}')


fig  = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

ax.scatter(x, Y)
ax.plot(x_space, Y_pred, 'r', label='Аналитическое решение')
ax.plot(x_space, Y_pred_gs, 'y', label='Градиентный спуск')
ax.set_xlabel('wavelength')
ax.set_ylabel('Spectr')
ax.legend()
ax.grid(True)
plt.show()