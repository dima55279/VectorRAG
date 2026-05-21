from tqdm import tqdm


class ProgressManager:

    @staticmethod
    def track(iterable, desc, total=None):

        return tqdm(
            iterable,
            desc=desc,
            total=total,
            ncols=100
        )