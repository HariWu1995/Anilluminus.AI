import os
from typing import Type

import onnxruntime as ort

from .models import sessions_class
from .models.base import BaseModel
from .models.u2net import U2net


def new_session(
    model_name: str = "u2net", providers=None, *args, **kwargs
) -> BaseModel:
    """
    Create a new session object based on the specified model name.

    This function searches for the session class based on the model name in the 'sessions_class' list.
    It then creates an instance of the session class with the provided arguments.
    The 'sess_opts' object is created using the 'ort.SessionOptions()' constructor.
    If the 'OMP_NUM_THREADS' environment variable is set, the 'inter_op_num_threads' option of 'sess_opts' is set to its value.

    Parameters:
        model_name (str): The name of the model.
        providers: The providers for the session.

           *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        BaseModel: The created session object.
    """
    session_class: Type[BaseModel] = U2net

    for sc in sessions_class:
        if sc.name() == model_name:
            session_class = sc
            break

    sess_opts = ort.SessionOptions()

    if "OMP_NUM_THREADS" in os.environ:
        sess_opts.inter_op_num_threads = int(os.environ["OMP_NUM_THREADS"])
        sess_opts.intra_op_num_threads = int(os.environ["OMP_NUM_THREADS"])

    return session_class(model_name, sess_opts, providers, *args, **kwargs)


