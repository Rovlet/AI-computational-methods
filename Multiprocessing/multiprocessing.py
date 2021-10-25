from multiprocessing import Process, Pipe, Pool
import os


def reverse(text):
    print(f'parent={os.getppid}, child={os.getpid}')
    return text[::-1]


def reverse2(pipe):
    while True:
        text = pipe.recv()
        pipe.send(text[::-1])


def mapper(line):
    return[(word.lower(), 1) for word in line.split() if word.isalppha]


if __name__ == '__main__':
    # zwykły przykłąd
    p = Process(target=reverse, args=('foobar',))
    p.start()
    p.join()

    # przykład z Pipe
    left, right = Pipe()
    Process(target=reverse2, args=(right,), daemon=True).start()
    left.send('hello')
    left.send('world')
    print(left.recv())
    print(left.recv())

    # przykład z Pool
    with open('file.txt') as big_data:
        with Pool() as pool:
            pool.map(mapper, big_data)
