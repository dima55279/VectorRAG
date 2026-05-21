from tqdm import tqdm


class ProgressManager:

    @staticmethod
    def create_progress(total, desc):

        return tqdm(
            total=total,
            desc=desc,
            ncols=100
        )