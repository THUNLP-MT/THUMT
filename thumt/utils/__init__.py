import glob

from thumt.utils.hparams import HParams
from thumt.utils.inference import beam_search


def latest_checkpoint(path):
    names = glob.glob(path + "/*.pt")

    if not names:
        return None

    latest_step = 0
    checkpoint_name = names[0]

    for name in names:
        step = name.rstrip(".pt").split("-")[-1]

        if not step.isdigit():
            continue
        else:
            step = int(step)

        if step > latest_step:
            checkpoint_name = name
            latest_step = step

    print("Latest checkpoint: %s" % checkpoint_name)

    return checkpoint_name
