import matplotlib.pyplot as plt
import numpy as np

def main():
    # Створюємо дані
    x = np.linspace(0, 10, 100) # 100 точок від 0 до 10
    y = np.sin(x) # Значення синуса для кожної точки x

    # Створюємо графік
    plt.figure(figsize=(8, 6)) # Розмір фігури 8x6 дюймів
    plt.plot(x, y, label='sin(x)', color='blue') # Графік синуса синього кольору
    plt.title('Графік функції sin(x)') # Заголовок графіка
    plt.xlabel('x') # Підпис осі x
    plt.ylabel('sin(x)') # Підпис осі y
    plt.grid(True) # Включаємо сітку
    plt.show()


if __name__ == '__main__':
    main()
