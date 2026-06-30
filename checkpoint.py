import numpy as np
class TinyValueNet:
    def __init__(self) -> None:
        self.w1 = np.zeros((16, 24), dtype=np.float32)
        self.b1 = np.zeros(24, dtype=np.float32)
        self.w2 = np.zeros((24, 1), dtype=np.float32)
        self.b2 = np.array([0.0], dtype=np.float32)

        primary = np.array(
            [
                1.10, 1.35, 3.25, -1.45, 0.70, 0.55, 0.65, 0.85,
                0.80, 0.45, 0.60, 0.70, 0.35, 0.40, 0.55, 0.25,
            ],
            dtype=np.float32,
        )
        secondary = primary * np.array(
            [
                0.55, 0.70, 0.80, 0.65, 0.60, 0.60, 0.55, 0.50,
                0.50, 0.55, 0.45, 0.45, 0.50, 0.40, 0.45, 0.35,
            ],
            dtype=np.float32,
        )

        for i in range(16):
            self.w1[i, i] = 2.25
            self.w1[i, i + 8] = 1.15
            self.w2[i, 0] = primary[i]
            self.w2[i + 8, 0] = secondary[i]

def main() -> None:
    model = TinyValueNet()
    np.savez(
        "hybrid_checkpoint.npz",
        w1=model.w1,
        b1=model.b1,
        w2=model.w2,
        b2=model.b2,
    )
    print("Saved hybrid_checkpoint.npz")

if __name__ == "__main__":
    main()
