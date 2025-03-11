import numpy as np

class GaussianAnimation():
    def __init__(self, w: int = 64, h: int = 64):
        self.w = w
        self.h = h
        self.mask = np.zeros((self.h, self.w))
        self.inc = np.random.random((self.h, self.w))
        self.dec = np.random.random((self.h, self.w))
    
    def _getPos(self):
        x = (np.ones((self.h, self.w)) * (np.linspace(0, self.w - 1, self.w) - self.w // 2))
        y = (np.ones((self.w, self.h)) * (np.linspace(0, self.h - 1, self.h) - self.h // 2)).T
        return x, y
    
    def maskCustom(self, matrix: np.ndarray):
        self.__init__(matrix.shape[1], matrix.shape[0])
        self.mask = matrix > np.mean(matrix)

    def maskCircle(self, radius: float):
        x, y = self._getPos()
        self.mask = np.where(x ** 2 + y ** 2 < radius ** 2, True, False)
        del x, y
    
    def next(self) -> np.ndarray:
        self.inc = np.concatenate([self.inc[1:], np.random.random((1, self.w))], 0)
        self.dec = np.concatenate([np.random.random((1, self.w)), self.dec[:-1]], 0)
        return np.where(self.mask, self.inc, self.dec)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.ion()
    anim = GaussianAnimation(256, 128)
    anim.maskCircle(50)
    for _ in range(100):
        plt.clf()
        plt.imshow(anim.next(), 'gray')
        plt.pause(.1)
