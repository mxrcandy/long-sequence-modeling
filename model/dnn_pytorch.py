from .base_model_pytorch import BaseModel


class DNN(BaseModel):
    def __init__(self,
                 *args,
                 long_seq_split=None,
                 **kwargs,
                 ):
        super(DNN, self).__init__(*args, **kwargs)