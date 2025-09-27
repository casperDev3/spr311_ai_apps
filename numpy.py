import numpy as np
import time


def main():
    # Створюємо масив для тестування
    size = 1_000_000
    arr_np_one = np.random.rand(size)
    arr_np_two = np.random.rand(size)

    # Рандомні 1 000 000 елементів звичайного масиву
    arr_one = [np.random.rand() for _ in range(size)]
    arr_two = [np.random.rand() for _ in range(size)]

    # Вимірюємо час для NumPy
    start_time = time.time()
    result_np = arr_np_one + arr_np_two
    numpy_time = time.time() - start_time
    print(f"NumPy time: {numpy_time:.6f} seconds")

    # Вимірюємо час для звичайного масиву
    start_time = time.time()
    result_list = [a + b for a, b in zip(arr_one, arr_two)]
    list_time = time.time() - start_time
    print(f"List time: {list_time:.6f} seconds")




if __name__ == '__main__':
    main()
