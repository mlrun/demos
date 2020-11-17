import abc


class AbsImage(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def convert_frame_to_bytes(self):
        pass

    @abc.abstractmethod
    def get_create_time(self):
        pass