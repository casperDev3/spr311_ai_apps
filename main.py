import subprocess  # Для додаткових операцій з процесами
import os  # Для роботи з операційною системою
import threading  # Для багатопоточності
import itertools  # Для ітераторів
import time  # Для роботи з часом
import sys  # Для взаємодії з інтерпретатором Python


def run_ollama(prompt, model="llama3"):
    """
    Викликає локальну модель Ollama з переданим текстом.
    """
    process = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    output, error = process.communicate(input=prompt)

    if error:
        print("⚠️ Помилка:", error)
    return output.strip()

def load_knowledge(base_dir="knowledge"):
    """
    Функція для завантаження знань з вказаної директорії.
    """
    knowledge_texts = []
    if os.path.exists(base_dir):
        for filename in os.listdir(base_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(base_dir, filename), 'r', encoding='utf-8') as file:
                    knowledge_texts.append(file.read())
    else:
        print(f"Directory {base_dir} does not exist.")
    return "\n\n".join(knowledge_texts)


def show_loading(stop_event):
    """
    Функція для відображення індикатора завантаження.
    """
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if stop_event.is_set():
            break
        sys.stdout.write('\rLoading ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rDone!     \n')


def main():
    print("Локальний чат-бот на основі Ollama")

    # Завантаження знань
    knowledge = load_knowledge()

    while True:
        user_input = input("Ти: ")
        if user_input.lower() in ['вихід', 'exit', 'quit']:
            print("Вихід з чат-бота.")
            break

        prompt = f"""
            Відповідай тільки українською мовою та емодзі у кожній відповіді.
            Ти — локальний AI. Використовуй за потреби ось ці знання для відповідей:
            {knowledge}
            
            Користувач запитує: {user_input}
        """

        # Запуск індикатора завантаження в окремому потоці
        stop_event = threading.Event()
        spinner = threading.Thread(target=show_loading, args=(stop_event,))
        spinner.start()

        # Отримання відповіді від моделі
        response = run_ollama(prompt, model="llama3")

        # Зупинка індикатора завантаження
        stop_event.set()
        spinner.join()

        print("Бот:", response)



if __name__ == '__main__':
    main()
