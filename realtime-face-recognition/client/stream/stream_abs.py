import abc


class AbsStream(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create_stream(self):
        pass

    @abc.abstractmethod
    def put_item(self, item):
        pass

    @abc.abstractmethod
    def get_item(self, item):
        pass
