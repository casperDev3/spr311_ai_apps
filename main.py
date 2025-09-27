import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def sin():
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

# sin()

def bar():
    data = {
        "Місяць": ['Січень', 'Лютий', 'Березень', 'Квітень', 'Травень'],
        "Продажі": [15000, 180000, 220000, 170000, 250000]
    }

    df = pd.DataFrame(data)

    # Створюємо стовпчасту діаграму
    plt.figure(figsize=(12, 6))
    # Створюємо стовпці
    bars = plt.bar(df['Місяць'], df['Продажі'], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height+500, f'int{height}', ha = 'center', va='bottom')

    plt.title('Продажі по місяцях', fontsize=16, fontweight="bold")
    plt.xlabel("Місяць")
    plt.ylabel('Продаіжі (грн)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

bar()