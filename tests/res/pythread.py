import logging
import threading
import time

G = 1

def thread_function(name):
    logging.info("Thread %s: starting %d", name, G)
    time.sleep(2)
    logging.info("Thread %s: finishing %d", name, G)

if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    logging.info("Main    : before creating thread %d", G)
    x = threading.Thread(target=thread_function, args=(1,))
    logging.info("Main    : before running thread %d", G)
    x.start()
    logging.info("Main    : wait for the thread to finish %d", G)
    G = 2
    # x.join()
    logging.info("Main    : all done %d", G)
