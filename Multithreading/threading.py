# Przykład fib

import threading


def fib(n):
    return n if n < 2 else fib(n - 2) + fib(n - 1)


if __name__ == '__main__':
    for _ in range(4):
        threading.Thread(target=fib, args=(50,)).start()

# Przykład z klasą

import threading
import time


class Task(threading.Thread):
    def run(self):
        try:
            time.sleep(1)
        finally:
            print("task done")


task = Task()
task.start()
task.join()  # bez tego print wykona się pierwszy
print("program finished")

task.daemon = True  # działanie w tle
exit()  # tu finally się nie wykona


# To samo funkcyjnie

def task():
    time.sleep(1)
    print("Task done")


threading.Thread(target=task, args=('foo',)).start()

# Symulacja wyścigu koni
import random


def sleep(name, message):
    time.sleep(random.random())
    print(name, message)


def horse(name):
    sleep(name, 'ready...')
    barier.wait()  # Żaden koń nie zacznie wyścigu za wcześnie
    sleep(name, 'started')
    sleep(name, 'finished')


def on_start():
    print('----Start----')


horse_names = ('Alfie', 'Daisy', 'Unity')
barier = threading.Barrier(len(horse_names), action=on_start)

horses = [
    threading.Thread(target=horse, args=(name,)) for name in horse_names
]

for horse in horses:
    horse.start()

for horse in horses:
    horse.join()


# Naciskanie klawisza jako start i stop
def on_key_press():
    while not finished.is_set():
        if key_pressed.wait(0.1):
            print('key pressed')
            key_pressed.clear()
        print('done')


key_pressed = threading.Event()
finished = threading.Event()
for _ in range(3):
    input()
    key_pressed.set()

threading.Thread(target=on_key_press).start()

# Synchronizacja dostępu
import random

request = threading.local()  # Dzięki temu pomimo że zmienna jest globalna to


#  w kontekście wątków będzie miała różne wartości
#  Nie używać w puli wątków

def controller():
    if request.method == 'GET':
        print(request.path)
    else:
        print('405 Method Not Allowed', request.method)


def dispatcher(http_request):
    request.method, request.path, _ = http_request.split()
    time.sleep(random.random())
    controller()


http_requests = [
    'PUT /resource/42 HTTP/1.1',
    'GET /resource/42 HTTP/1.1',
    'DELETE /resource/42 HTTP/1.1'
]

for http_request in http_requests:
    threading.Thread(target=dispatcher, args=(http_request,)).start()

# Wypłata

balance = 80
balance_lock = threading.Lock()  # wada - mała przepustowość, ryzyko wystąpienia zakleszczenia


def withdraw(amount, person):
    print(f'{person} wants to withdraw ${amount}\n')
    global balance
    with balance_lock:
        if balance - amount >= 0:
            time.sleep(1)
            balance -= amount
            print(f'-${amount} accepted')
        else:
            print(f'-${amount} rejected')


t1 = threading.Thread(target=withdraw, args=(80, 'Alice'))
t2 = threading.Thread(target=withdraw, args=(50, 'Bob'))

t1.start()
t2.start()

t1.join()
t2.join()

print(balance)

# Zakleszczenie

lock = threading.Lock()

print(lock.acquire(blocking=False))  # True
print(lock.acquire(blocking=False))  # False

# RLock
lock = threading.RLock()

print(lock.acquire(blocking=False))  # True
print(lock.acquire(blocking=False))  # True

# Problem konsumenta i producenta

import logging
import random
import collections

logging.basicConfig(level=logging.INFO, format='%(message)s')


class SynchronizedBuffer:
    def __init__(self, capacity):
        self.values = collections.deque(maxlen=capacity)
        self.lock = threading.RLock()
        self.consumed = threading.Condition(self.lock)
        self.produced = threading.Condition(self.lock)

    def __repr__(self):
        return repr(list(self.values))

    def is_empty(self):
        return len(self.values) == 0

    def is_full(self):
        return len(self.values) == self.values.maxlen

    def put(self, value):
        with self.lock:
            while self.is_full():
                self.consumed.wait()
            self.values.append(value)
            self.produced.notify()  # wybudza 1 wątek konsumenta, nie zwalnia automatycznie blokady

    def get(self):
        with self.lock:
            while self.is_empty():
                self.produced.wait()
            try:
                return self.values.popleft()
            finally:
                self.consumed.notify()


def producer(buffer):
    while True:
        buffer.put(random.randint(1, 10))
        logging.info('produces %s', buffer)
        time.sleep(random.random())


def consumer(buffer):
    while True:
        buffer.get()
        logging.info('consumed %s', buffer)
        time.sleep(random.random())


buffer = SynchronizedBuffer(5)

for _ in range(3):
    threading.Thread(target=producer, args=(buffer,)).start()

for _ in range(2):
    threading.Thread(target=consumer, args=(buffer,)).start()
