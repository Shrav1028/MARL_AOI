class AOIPacket:
    def __init__(self, start_step):
        self.start_step = start_step

    def get_age(self, current_step):
        return current_step - self.start_step

